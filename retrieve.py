from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document
from config import TABLE_NAME

def get_relevant_documents(query: str, vector_stored: PostgresVectorStore) -> list[Document]:
    """
    Récupère les documents les plus pertinents pour une requête donnée en utilisant MMR.
    """
    retriever = vector_stored.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'lambda_mult': 0.25}
    )
    documents_with_scores = retriever.invoke(query)
    return documents_with_scores

def format_relevant_documents(documents: list[Document]) -> str:
    """
    Formate les documents pertinents pour affichage.
    """
    return "\n-----\n".join([
        f"Source {i + 1}: {doc.page_content}\nScore: {getattr(doc, 'score', 'N/A')} (mmr)"
        for i, doc in enumerate(documents)
    ])

if __name__ == '__main__':
    Example_de_test = """
    What are the differences between LSTM and GRU in handling vanishing gradients?
    How does the self-attention mechanism work in transformers? 
    Looking for examples and formulas for understanding attention scores.
    """
    engine = create_cloud_sql_database_connection()
    embedding = get_embeddings()
    vector_store = get_vector_store(engine, TABLE_NAME, embedding)
    documents = get_relevant_documents(Example_de_test, vector_store)
    assert len(documents) > 0, "No documents found for the query"
    doc_str: str = format_relevant_documents(documents)
    assert len(doc_str) > 0, "No documents formatted successfully"
    print("All tests passed successfully.")
