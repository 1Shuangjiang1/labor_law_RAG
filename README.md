



# 📚 RAG Demo for Labor Law QA

This is a simple Retrieval-Augmented Generation (RAG) demo designed for labor law question-answering. It combines document retrieval, embedding, and large language models (LLMs) to answer user questions via a Gradio web interface.

👉 For detailed technical explanation and insights, please refer to my blog: [Your Blog Link Here](https://1shuangjiang1.github.io/p/%E5%9F%BA%E4%BA%8Ellamaindex%E7%9A%84%E5%8A%B3%E5%8A%A8%E6%B3%95rag%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F/)

---

## 📂 Project Structure

```bash
├── download.py       # Downloads embedding and LLM models (e.g., from ModelScope)
├── embedding.py      # Embeds documents and saves the vector index locally
├── get_data.py       # Scrapes labor law content from a specified website
├── RAG.py            # Full RAG pipeline with a Gradio QA interface
├── requirements.txt  # Python dependencies
````



## 🚀 How to Use

1. **Download Models**

   ```bash
   python download.py
   ```

   > ✏️ You need to manually specify the model names and save paths in the script (supports downloading from ModelScope or other sources).

2. **Download Legal Documents**

   ```bash
   python get_data.py
   ```

   > The script includes a pre-filled labor law URL. You can modify it as needed.

3. **Generate Vector Embeddings**

   ```bash
   python embedding.py
   ```

   > Fill in your local embedding model path in the script. The output will be a vectorized index of your document collection.

4. **Run the Full RAG Pipeline (with Gradio UI)**

   ```bash
   python RAG.py
   ```

   > This script integrates all components—retrieval, generation, and UI—into one pipeline. It launches a Gradio web app for interactive QA.

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

It's recommended to use a virtual environment (e.g., `venv` or `conda`).

---

## 📝 Notes

* Please manually configure model paths and download options in the scripts.
* This project is intended for educational or experimental use, not optimized for production.
* Supports local embedding + LLM models; Gradio is used for easy prototyping.

---

## 👨‍💻 Author

* AI undergraduate student from South China University of Technology
* Passionate about LLM applications, RAG systems, and AI engineering

Feel free to open issues or reach out if you have suggestions or questions!

```


