from  pymilvus import utility, Collection, RRFRanker, AnnSearchRequest
import asyncio
from typing import List, Optional
import requests
import time

# Singleton instances
_reranker = None
_embedding_llama_server_url = "http://localhost:8080/embedding"
_init_done = False
_collection = None

COLLECTION_NAME = 'asr'
RRF_K = 70  # Reciprocal Rank Fusion parameter
QWEN_EMBED_DIM = 4096

DOC_INSTRUCTION = "Instruct: Represent this video for retrieval, focusing on the main events, key entities (people, places, organizations), and factual details.\nNews report: "
QUERY_INSTRUCTION = "Instruct: Retrieve the video segment that relates to the following topic or event from the query.\nQuery: "

def llama_server_encode_documents(
    texts: List[str], 
    server_url: str = "http://localhost:8080/embedding",
    batch_size: int = 32,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    doc_instruction: Optional[str] = None
) -> List[List[float]]:
    """
    Generate embeddings for texts using llama-server instead of local embedding model.
    
    Args:
        texts: List of strings to encode
        server_url: URL of the llama-server embedding endpoint
        batch_size: Number of texts to process in each batch
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Delay between retries in seconds
        doc_instruction: Optional instruction to prepend to documents
        
    Returns:
        List of embedding vectors corresponding to input texts
    """
    all_embeddings = []
    
    # Process in batches to avoid overwhelming the server
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        if doc_instruction:
            # Prepend instruction to each text if provided
            batch = [f"{doc_instruction} {text}" for text in batch]
        
        # Prepare request payload
        payload = {"content": batch}
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                # Send request to llama-server
                response = requests.post(server_url, json=payload, timeout=30)
                response.raise_for_status()
                
                # Extract embeddings from response
                result = response.json()
                # result is a list of objects with 'index' and 'embedding'
                # Sort by 'index' to ensure correct order
                batch_embeddings = [None] * len(batch)
                for obj in result:
                    idx = obj["index"]
                    batch_embeddings[idx] = obj["embedding"][0]


                if len(batch_embeddings) != len(batch):
                    raise ValueError(f"Expected {len(batch)} embeddings but got {len(batch_embeddings)}")
                
                # Add embeddings to result list
                all_embeddings.extend(batch_embeddings)
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt+1}/{max_retries}: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to encode batch after {max_retries} attempts: {str(e)}")
                    # Return empty vectors as fallback
                    empty_vectors = [[0.0] * QWEN_EMBED_DIM for _ in batch]
                    all_embeddings.extend(empty_vectors)
    
    return all_embeddings

def encode_queries(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    return llama_server_encode_documents(
        texts,
        batch_size=batch_size,
        server_url=_embedding_llama_server_url,
        doc_instruction=QUERY_INSTRUCTION
    )

def _init_sync(embedding_llama_server_url: str = "http://localhost:8080/embedding"):
    """
    Synchronous initialization of embedding function and reranker.
    """
    global _reranker, _init_done, _embedding_llama_server_url, _collection
    if not _init_done:
        if not utility.has_collection(COLLECTION_NAME):
            raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")
        
        _reranker = RRFRanker(RRF_K)
        _embedding_llama_server_url = embedding_llama_server_url
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        _init_done = True

async def init(embedding_llama_server_url: str = "http://localhost:8080/embedding"):
    """
    Async initialization wrapper to avoid blocking startup.
    """
    await asyncio.to_thread(_init_sync, embedding_llama_server_url)

def search_by_asr(
    query: str,
    top_k: int = 100,
    subset: list[str] = None
):
    assert isinstance(query, str), "Query must be a string"
    assert isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer"
    assert len(query) > 0, "Query cannot be empty"

    if not _init_done:
        raise RuntimeError("Model is not initialized. Call init() before using search_by_asr.")
    
    if not query:
        raise ValueError("Query cannot be empty.")

    query_vector = encode_queries([query])

    # Build filter expression if subset is provided
    filter_expr = None
    if subset is not None and len(subset) > 0:
        # Create filter conditions for ids starting with any value in subset
        # Milvus supports LIKE operator for string pattern matching
        conditions = []
        for prefix in subset:
            # Escape double quotes in prefix to prevent issues
            escaped_prefix = prefix.replace('"', '\\"')
            conditions.append(f'audio_segment_id LIKE "{escaped_prefix}%"')
        filter_expr = " OR ".join(conditions)

    dense_search_params = {
        "data": query_vector,
        "anns_field": "qwen_embed_vector",
        "param": {
            "ef": 4 * top_k,
        },
        "limit": top_k,
    }
    
    # Add filter expression if provided
    if filter_expr:
        dense_search_params["expr"] = filter_expr

    dense_request = AnnSearchRequest(**dense_search_params)

    sparse_search_params = {
        "data": [query],
        "anns_field": "sparse_vector",
        "param": {
            "drop_ratio_search": 0.2
        },
        "limit": top_k,
    }
    
    # Add filter expression if provided
    if filter_expr:
        sparse_search_params["expr"] = filter_expr

    sparse_request = AnnSearchRequest(**sparse_search_params)

    result = _collection.hybrid_search(
        reqs=[dense_request, sparse_request],
        rerank=_reranker,
        limit=top_k,
        output_fields=["audio_segment_id", "keyframes"]
    )[0]

    # collect keyframe+distance pairs
    keyframes_list: list[tuple[str, float]] = []
    for res in result:
        distance = res.distance
        keyframes = res.entity.get("keyframes", [])
        for keyframe in keyframes:
            keyframes_list.append((keyframe, distance))

    # sort by ascending distance (higher distance = higher rank)
    keyframes_list.sort(key=lambda x: x[1], reverse=True)

    # apply Reciprocal Rank Fusion (RRF)
    rrf_buckets: dict[str, list[int]] = {}
    fused_list: list[tuple[str, float]] = []
    last_distance = None
    last_rank = 0
    for _, (keyframe, distance) in enumerate(keyframes_list):
        if distance != last_distance:
            rank = last_rank + 1
        else:
            rank = last_rank
        last_distance = distance
        last_rank = rank
        rrf_buckets.setdefault(keyframe, []).append(rank)

    # compute and sort by fused RRF score
    for keyframe, ranks in rrf_buckets.items():
        score = sum(1.0 / (RRF_K + r) for r in ranks)
        fused_list.append((keyframe, score))
    fused_list.sort(key=lambda x: x[1], reverse=True)

    return fused_list[:top_k]  # Return only the top_k results
 
def _cleanup_sync():
    """
    Synchronous cleanup of embedding function and reranker.
    """
    global _reranker, _init_done, _collection
    # Dereference resources
    _reranker = None
    _collection.release()
    _collection = None
    _init_done = False

async def shutdown():
    """
    Async cleanup wrapper to cleanup embedding function and associated resources.
    """
    await asyncio.to_thread(_cleanup_sync)
