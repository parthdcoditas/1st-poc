from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from embedding_generate import similar_document, db_config, embeddings_model
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

connection = psycopg2.connect(
    dbname=db_config["dbname"],
    user=db_config["user"],
    password=db_config["password"],
    host=db_config["host"],
    port=db_config["port"],
)
cursor = connection.cursor()

class QueryState(BaseModel):
    user_query: str = Field(default="")
    embedding_list: list = Field(default_factory=list)
    cached: bool = Field(default=False)
    cached_query: str = Field(default="")
    response: str = Field(default="")
    context: str = Field(default="")
    force_llm_call: bool = Field(default=False) 

def cache_search(state: QueryState):
    state.embedding_list = np.array(embeddings_model.embed_query(state.user_query)).tolist()
    SIMILARITY_THRESHOLD = 0.5
    cursor.execute(
        """
        SELECT response, user_query, 1 - (embedding <-> %s::vector) AS similarity
        FROM cache_storage
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (state.embedding_list, state.embedding_list, 5),
    )
    results = cursor.fetchall()
    if results:
        best_response, cached_query, max_similarity = results[0]
        if max_similarity >= SIMILARITY_THRESHOLD:
            state.cached = True
            state.response = best_response
            state.cached_query = cached_query
    return state

def get_context_and_generate_response(state: QueryState):
    """Generate a response using the LLM if no cached response is found."""
    if state.cached and not state.force_llm_call:
        return state  

    retriever = similar_document()
    similar_docs = retriever.invoke(state.user_query)
    state.context = "\n".join([doc.page_content for doc in similar_docs])

    prompt = PromptTemplate(
        input_variables=["query", "context"], 
        template="Given the following context: {context}\nAnswer the query in 2-3 sentences: {query}"
    )
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768")

    llm_chain = prompt | llm

    response = llm_chain.invoke({"query": state.user_query, "context": state.context})
    state.response = response.content if hasattr(response, 'content') else str(response)
    
    return state

def store_query_and_response(state: QueryState):
    if state.cached_query:
        cursor.execute(
            """
            UPDATE cache_storage
            SET user_query = %s, response = %s, embedding = %s::vector
            WHERE user_query = %s;
            """,
            (state.user_query, state.response, state.embedding_list, state.cached_query),
        )
    else:
        cursor.execute(
            """
            INSERT INTO cache_storage (user_query, embedding, response)
            VALUES (%s, %s::vector, %s);
            """,
            (state.user_query, state.embedding_list, state.response),
        )

    connection.commit()
    return state

workflow = StateGraph(QueryState)

workflow.add_node("cache_search", cache_search)
workflow.add_node("get_context_and_generate_response", get_context_and_generate_response)
workflow.add_node("store_query_and_response", store_query_and_response)

workflow.add_edge(START, "cache_search")
workflow.add_edge("cache_search", "get_context_and_generate_response")
workflow.add_edge("get_context_and_generate_response", "store_query_and_response")
workflow.add_edge("store_query_and_response", END)

compiled_app = workflow.compile()
