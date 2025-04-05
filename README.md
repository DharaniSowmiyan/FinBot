# 💸 FinBot — AI-Powered Financial Chatbot

FinBot is an intelligent financial assistant that leverages **retrieval-augmented generation (RAG)** to answer finance-related queries based on uploaded PDF reports. It combines powerful embeddings with natural language generation to provide accurate and conversational responses.

---

## 🚀 Features

- 🧠 RAG-based architecture using LlamaIndex + ChromaDB  
- 📄 Ingests and indexes financial PDF documents  
- 🔍 Semantic search using HuggingFace embeddings  
- 🗣️ Answers your financial queries using Zephyr-7B model  
- 💡 Built with Python, Streamlit, Transformers, and more

---

## 📂 Project Structure

```
FinBot/
│
├── files/                   # PDF files to ingest
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # You're reading it now!
└── .gitignore              # Files to ignore in Git
```

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/FinBot.git
   cd FinBot
   ```

2. **Create a virtual environment** (optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF reports**  
   Place your `.pdf` files in the `files/` folder.

5. **Run the chatbot**  
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Tech Stack

- Python 🐍  
- Streamlit  
- HuggingFace Transformers  
- Zephyr-7B  
- LlamaIndex  
- ChromaDB  
- PyMuPDF (fitz)  
- scikit-learn, numpy

---

## 🗣 Example Query

> "Summarize the key financial metrics from Q4 report."

> "What were the revenue highlights of Company X in 2023?"

---

## 🙌 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/)  
- [ChromaDB](https://www.trychroma.com/)  
- [Hugging Face](https://huggingface.co/)


