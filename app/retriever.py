from .embedder import get_embedding

def retrieve_relevant_chunks(query: str, vector_store, top_k=5):
    """Retrieve relevant chunks from vector store based on query"""
    query_emb = get_embedding(query)
    results = vector_store.search(query_emb, top_k=top_k)
    return [chunk for chunk, _ in results]