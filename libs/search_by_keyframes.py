from pymilvus import utility, Collection
import torch
import asyncio
from eva_clip import create_model, get_tokenizer
import numpy as np

# Singleton instances for model, preprocessor, and reranker
_device = None
_model = None
_tokenizer = None
_init_done = False
_collection = None

COLLECTION_NAME = 'keyframes'

MODEL_NAME = "EVA02-CLIP-L-14-336" 
PATH_TO_WEIGHT = "./assets/EVA02_CLIP_L_336_psz14_s6B.pt" # From root of the project

def _init_sync(device: str = 'cpu'):
    """
    Synchronous initialization of model, preprocessor, and reranker.
    """
    global _device, _model, _tokenizer, _init_done, _collection
    if not _init_done:
        if not utility.has_collection(COLLECTION_NAME):
            raise Exception(f"Collection '{COLLECTION_NAME}' does not exist. Please create it first.")

        _device = device
        _collection = Collection(COLLECTION_NAME)
        _collection.load()

        _model = create_model(MODEL_NAME, PATH_TO_WEIGHT, force_custom_clip=True)
        if _device != 'cpu':
            _model = _model.to(_device)
        _model.eval()

        _tokenizer = get_tokenizer(MODEL_NAME)

        _init_done = True

async def init(device: str = 'cpu'):
    """
    Async initialization wrapper to avoid blocking startup.
    """
    await asyncio.to_thread(_init_sync, device)

def search_by_keyframes(
    image_id_query=None,
    text_query=None,
    top_k=5,
    subset: list[str] = None
):
    """
    Searches for keyframes in the Milvus collection.

    Args:
        image_id_query (str): The ID of the image to search for.
        text_query (str): The text query to search for.
        top_k (int): The maximum number of results to return.
        subset (list[str]): The subset of keyframes to search for.
    Returns:
        A list of dictionaries containing the search results.
        Returns an error message if the collection doesn't exist.
    """
    if not _init_done:
        raise RuntimeError("Model is not initialized. Call init() before using search_keyframes.")

    # Build filter expression if subset is provided
    filter_expr = None
    if subset is not None and len(subset) > 0:
        # Create filter conditions for ids starting with any value in subset
        # Milvus supports LIKE operator for string pattern matching
        conditions = []
        for prefix in subset:
            # Escape double quotes in prefix to prevent issues
            escaped_prefix = prefix.replace('"', '\\"')
            conditions.append(f'keyframe_id LIKE "{escaped_prefix}%"')
        filter_expr = " OR ".join(conditions)

    nprobe = None

    if top_k <= 256:
        nprobe = 16
    elif top_k <= 512:
        nprobe = 20
    else :
        nprobe = 32

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": nprobe
        }
    }

    with torch.no_grad(), torch.amp.autocast(_device):
        if image_id_query:
            """For KNN query"""
            _res = _collection.query(
                expr=f"keyframe_id == '{image_id_query}'",
                limit=1,
                output_fields=["eva_clip_vector"],
            )
            query_vector = np.frombuffer(_res[0]['eva_clip_vector'][0], dtype=np.float16).reshape((1, -1))
        elif text_query:
            tensor = _tokenizer([text_query]).to(_device)
            query_vector = _model.encode_text(tensor)
            query_vector /= query_vector.norm(dim=-1, keepdim=True)
            query_vector = query_vector.half().cpu().numpy()
        else:
            raise ValueError("Either image_id_query or text_query must be provided.")
        
    result = _collection.search(
        data=query_vector,
        anns_field="eva_clip_vector",
        param=search_params,
        limit=top_k,
        output_fields=["keyframe_id"],
        expr=filter_expr
    )[0]

    keyframes_list: list[tuple[str, float]] = []
    for res in result:
        distance = res.distance
        keyframe_id = res.entity.get("keyframe_id")
        keyframes_list.append((keyframe_id, distance))

    keyframes_list.sort(key=lambda x: x[1], reverse=True)  # Sort by distance

    return keyframes_list

def _cleanup_sync():
    """
    Synchronous cleanup of model, preprocessor, and reranker.
    """
    global _device, _model, _tokenizer, _init_done, _collection

    _model = None
    _tokenizer = None

    if _device is not None and _device != 'cpu':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _device = None
    _collection.release()
    _collection = None
    _init_done = False

async def shutdown():
    """
    Async cleanup wrapper to cleanup model and associated resources.
    """
    await asyncio.to_thread(_cleanup_sync)
