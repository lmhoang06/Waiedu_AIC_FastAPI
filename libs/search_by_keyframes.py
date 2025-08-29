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

COLLECTION_NAME = 'keyframes'

MODEL_NAME = "EVA02-CLIP-L-14-336" 
PATH_TO_WEIGHT = "./assets/EVA02_CLIP_L_336_psz14_s6B.pt" # From root of the project

def _init_sync(device: str = 'cpu'):
    """
    Synchronous initialization of model, preprocessor, and reranker.
    """
    global _device, _model, _tokenizer, _init_done
    if not _init_done:
        _device = device

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
    top_k=5
):
    """
    Searches for keyframes in the Milvus collection.

    Args:
        image_id_query (str): The ID of the image to search for.
        text_query (str): The text query to search for.
        top_k (int): The maximum number of results to return.

    Returns:
        A list of dictionaries containing the search results.
        Returns an error message if the collection doesn't exist.
    """
    if not utility.has_collection(COLLECTION_NAME):
        raise Exception(f"Collection '{COLLECTION_NAME}' does not exist. Please create it first.")
    
    if not _init_done:
        raise RuntimeError("Model is not initialized. Call init() before using search_keyframes.")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    nprobe = None

    if top_k <= 256:
        nprobe = 8
    elif top_k <= 512:
        nprobe = 16
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
            _res = collection.query(
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
        
    result = collection.search(
        data=query_vector,
        anns_field="eva_clip_vector",
        param=search_params,
        limit=top_k,
        output_fields=["keyframe_id"]
    )[0]

    collection.release()

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
    global _device, _model, _tokenizer, _init_done

    _model = None
    _tokenizer = None

    if _device is not None and _device != 'cpu':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _device = None
    _init_done = False

async def shutdown():
    """
    Async cleanup wrapper to cleanup model and associated resources.
    """
    await asyncio.to_thread(_cleanup_sync)
