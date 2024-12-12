from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import os
db_config = {
    "dbname": os.getenv("dbname"),
    "user": os.getenv("username"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port")
}
connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PGVector(
    embeddings=embeddings_model,
    collection_name="pdf_texts",  
    connection=connection_string,  
    use_jsonb=True,
)

def generate_embeddings(text_chunks):
    for chunk in text_chunks:
        vector_store.add_texts([chunk])

def similar_document():
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

