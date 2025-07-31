from .embedder import get_embedding

def retrieve_relevant_chunks(query: str, vector_store, top_k=5):
    query_emb = get_embedding(query)
    results = vector_store.search(query_emb, top_k=top_k)
    print(f"[DEBUG] Question: {query}")
    print(f"[DEBUG] Top retrieved chunks:")
    for i, (chunk, score) in enumerate(results):
        print(f"  Chunk {i} (score {score:.2f}): {chunk[:200]}")
    return [chunk for chunk, _ in results]