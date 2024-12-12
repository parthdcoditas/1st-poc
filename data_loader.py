import os
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_generate import generate_embeddings

directory = './PDF'

def get_pdf_file_paths(directory):
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith('.pdf')
    ]

def load_text_samples():
    file_paths = get_pdf_file_paths(directory)
    texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        for split in splits:
            content = split.page_content.strip()
            if content:
                texts.append(content)
    return texts

text_chunk = load_text_samples()
generate_embeddings(text_chunk)
