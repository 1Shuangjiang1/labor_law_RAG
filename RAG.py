import gradio as gr
import json
import time
from pathlib import Path
from typing import List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate

# 设置查询模板
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)


# 配置模型路径
class Config:
    EMBED_MODEL_PATH = r"C:\hugging_face\GTE"
    LLM_MODEL_PATH = r"C:\hugging_face\Qwen3_1.7B"

    DATA_DIR = "data"
    VECTOR_DB_DIR = "chroma_db"
    PERSIST_DIR = "storage"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 3


# 初始化模型
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
        trust_remote_code=True
    )
    llm = HuggingFaceLLM(
        model_name=Config.LLM_MODEL_PATH,
        tokenizer_name=Config.LLM_MODEL_PATH,
        model_kwargs={"trust_remote_code": True,
                      "torch_dtype": "auto"},
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3}
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")

    return embed_model, llm


# 加载并验证JSON数据
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                                    "content": item,
                                    "metadata": {"source": json_file.name}
                                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")

    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data


# 创建文本节点
def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]

        for full_title, content in law_dict.items():
            node_id = f"{source_file}::{full_title}"
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"

            node = TextNode(
                text=content,
                id_=node_id,
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)

    return nodes


# 初始化向量存储
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if chroma_collection.count() == 0 and nodes is not None:
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    return index


# 创建查询引擎
def create_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        verbose=True
    )


# 主要功能：问题查询和响应
def query_law(question: str):
    embed_model, llm = init_models()

    if not Path(Config.VECTOR_DB_DIR).exists():
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        nodes = None

    index = init_vector_store(nodes)
    query_engine = create_query_engine(index)

    response = query_engine.query(question)

    # 提取模型的回答
    assistant_answer = response.response

    # 提取支持文档的内容并转化为元组形式
    supporting_documents = []
    for idx, node in enumerate(response.source_nodes, 1):
        meta = node.metadata
        supporting_documents.append((
            meta['full_title'],  # 法律条款
            meta['source_file'],  # 来源文件
            meta['law_name'],  # 法律名称
            node.text[:100],  # 条款内容（仅显示前100个字符）
            node.score  # 相关度得分
        ))

    # 返回两个独立的输出
    return assistant_answer, supporting_documents


# 使用 Gradio 创建UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 劳动法法律查询助手")

        # 输入框
        question_input = gr.Textbox(label="请输入劳动法相关问题", placeholder="请输入问题并提交", lines=2)

        # 查询按钮
        submit_btn = gr.Button("查询")

        # 响应框
        response_output = gr.Textbox(label="智能助手回答", lines=5)

        # 文档显示框
        documents_output = gr.DataFrame(headers=["法律条款", "来源文件", "法律名称", "条款内容", "相关度得分"],
                                        show_label=True)

        # 设置按钮事件
        submit_btn.click(query_law, inputs=question_input, outputs=[response_output, documents_output])

    demo.launch()


if __name__ == "__main__":
    main()
