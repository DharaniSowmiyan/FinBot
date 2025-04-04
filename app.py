import os
import torch
import chromadb
import fitz
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize ChromaDB client (persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")
vector_store = ChromaVectorStore(chroma_client=client)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# Load PDF documents
pdf_directory = "./files"
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        documents.append(Document(text=text))  # Convert to Document objects

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create LlamaIndex
index_store_path = "./index_store"

if os.path.exists(index_store_path):
    print("🔄 Reloading existing index...")
    index = VectorStoreIndex.from_persist_dir(index_store_path, embed_model=embed_model)
else:
    print("🆕 Creating a new index...")
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embed_model)
    index.storage_context.persist(persist_dir=index_store_path)  # Save index

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModel.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

# Initialize text generation pipeline
generator_pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Function to retrieve similar documents
def retrieve_similar_documents(query, index, top_k=5):
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    
    # Extract text from retrieved response
    retrieved_docs = [doc.text for doc in response.source_nodes] if response.source_nodes else []
    
    return retrieved_docs

# Function to handle RAG-based retrieval and generation
def rag_query_with_generation(query, index, generator_pipe, top_k=5):
    retrieved_docs = retrieve_similar_documents(query, index, top_k)
    context_text = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
    
    combined_prompt = f"Context: {context_text}\nQuery: {query}\nAnswer:"
    
    generated_text = generator_pipe(
        combined_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )[0]["generated_text"]
    
    return generated_text

# Streamlit chatbot UI
def fintech_chatbot_app():
    st.set_page_config(page_title="Fintech Chatbot", page_icon="💸", layout="wide")
    st.title("AI-powered Financial Chatbot")
    st.image("background_image.png", width=150)
    
    user_input = st.text_input("You:", "")
    if st.button("Send"):
        response = rag_query_with_generation(user_input, index, generator_pipe)
        st.text_area("Bot:", value=response, height=100, key="output")

if __name__ == "__main__":
    fintech_chatbot_app() 