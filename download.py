#模型下载
from modelscope import snapshot_download
# 下载 DeepSeek-R1-Distill-Qwen-1.5B 模型,并下载到指定路径
def download_model(model_name: str, local_dir: str):
    """下载指定的模型到本地目录"""
    snapshot_download(
        model_id=model_name,
        local_dir=local_dir,
    )
    print(f"模型 {model_name} 已下载到 {local_dir}")

if __name__ == '__main__':
    # 模型名称和本地目录
    model_name = "Qwen/Qwen3-1.7B"
    local_dir = "C:\hugging_face\Qwen3_1.7B"

    # 下载模型
    download_model(model_name, local_dir)