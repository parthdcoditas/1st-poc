from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from embedding_generate import similar_document, db_config, embeddings_model
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

connection = psycopg2.connect(
    dbname=db_config["dbname"],
    user=db_config["user"],
    password=db_config["password"],
    host=db_config["host"],
    port=db_config["port"],
)
cursor = connection.cursor()

def cache_search(user_query):
    """Search for a cached response."""
    user_query_embedding = embeddings_model.embed_query(user_query)
    embedding_list = np.array(user_query_embedding).tolist()

    SIMILARITY_THRESHOLD = 0.5  

    cursor.execute(
        """
        SELECT response, user_query, 1 - (embedding <-> %s::vector) AS similarity
        FROM cache_storage
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (embedding_list, embedding_list, 5),
    )

    results = cursor.fetchall()

    if results:
        best_response, cached_query, max_similarity = results[0]
        if max_similarity >= SIMILARITY_THRESHOLD:
            return {"cached": True, "response": best_response, "cached_query": cached_query}

    new_response = get_context_and_generate_response(user_query, embedding_list)
    return {"cached": False, "response": new_response}

def get_context_and_generate_response(user_query, embedding_list, cached_query=None):
    retriever = similar_document()
    similar_docs = retriever.invoke(user_query)

    context = "\n".join([doc.page_content for doc in similar_docs])

    prompt = PromptTemplate(
        input_variables=["query", "context"], 
        template="Given the following context: {context}\nAnswer the query in 2-3 sentences: {query}"
    )
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768")

    llm_chain = prompt | llm

    response = llm_chain.invoke({"query": user_query, "context": context})
    response_content = response.content if hasattr(response, 'content') else str(response)

    store_query_and_response(user_query, embedding_list, response_content, cached_query)

    return response_content

def store_query_and_response(user_query, embedding_list, llm_response, cached_query=None):
    llm_response = str(llm_response)

    if cached_query:
        cursor.execute(
            """
            UPDATE cache_storage
            SET user_query = %s, response = %s, embedding = %s::vector
            WHERE user_query = %s;
            """,
            (user_query, llm_response, embedding_list, cached_query),
        )
    else:
        cursor.execute(
            """
            INSERT INTO cache_storage (user_query, embedding, response)
            VALUES (%s, %s::vector, %s);
            """,
            (user_query, embedding_list, llm_response),
        )

    connection.commit()
