from pymilvus import utility, Collection
from transformers import AutoConfig, AutoModel, AutoProcessor
import torch
import asyncio

# Singleton instances for model, preprocessor, and reranker
_device = None
_config = None
_model = None
_preprocess = None
_dtype = None

_init_done = False

COLLECTION_NAME = 'scenes'

def _init_sync(device: str = 'cpu'):
    """
    Synchronous initialization of model, preprocessor, and reranker.
    """
    global _device, _config, _model, _preprocess, _dtype, _init_done
    if not _init_done:
        _device = device
        path = "nvidia/Cosmos-Embed1-448p"
        _config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        _model = AutoModel.from_pretrained(path, trust_remote_code=True, config=_config)
        
        # Determine dtype based on device and CUDA compute capability
        if _device == 'cpu':
            _dtype = torch.float32
        else:
            torch_device = torch.device(_device)
            if torch_device.type == 'cuda':
                major, _ = torch.cuda.get_device_capability(torch_device)
                if major >= 8:  # Ampere and newer support bfloat16
                    _dtype = torch.bfloat16
                else:
                    _dtype = torch.float32
            else:
                _dtype = torch.float32  # Fallback for non-CUDA accelerators
        
        _model = _model.to(_device, dtype=_dtype)
        _model.eval()

        print("Cosmos-embed1 is initialized with dtype =", _dtype)

        from multiprocessing import set_start_method
        try:
            set_start_method("fork")
        except RuntimeError:
            pass # start method can only be set once
        _preprocess = AutoProcessor.from_pretrained(path, trust_remote_code=True)

        _init_done = True

async def init(device: str = 'cpu'):
    """
    Async initialization wrapper to avoid blocking startup.
    """
    await asyncio.to_thread(_init_sync, device)

def search_by_scenes(
    query: str,
    top_k: int = 100,
):
    assert isinstance(query, str), "Query must be a string"
    assert isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer"
    assert len(query) > 0, "Query cannot be empty"

    if not _init_done:
        raise RuntimeError("Model is not initialized. Call init() before using search_by_asr.")
    
    if not query:
        raise ValueError("Query cannot be empty.")
    
    # Assuming initialization done at startup; no sync init call here
    if not utility.has_collection(COLLECTION_NAME):
        raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")
    
    collection = Collection(COLLECTION_NAME)
    collection.load()

    with torch.no_grad():
        text_inputs = _preprocess(text=query).to(_device, dtype=_dtype)
        query_vector = _model.get_text_embeddings(**text_inputs).text_proj.cpu().numpy()

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "ef": 4 * top_k
        }
    }

    result = collection.search(
        data=query_vector,
        anns_field="cosmos_embed_vector",
        param=search_params,
        limit=top_k,
        output_fields=["scene_id", "keyframes", "start_frame", "end_frame"]
    )[0]

    collection.release()

    keyframes_list: list[tuple[str, float]] = []
    for res in result:
        distance = res.distance
        keyframes = res.entity.get("keyframes", [])
        for keyframe in keyframes:
            keyframes_list.append((keyframe, distance))

    keyframes_list.sort(key=lambda x: x[1], reverse=True)  # Sort by distance

    return keyframes_list[:top_k]

def _cleanup_sync():
    """
    Synchronous cleanup of model, preprocessor, and reranker.
    """
    global _device, _config, _model, _preprocess, _dtype, _init_done
    
    _model = None
    _preprocess = None
    _config = None
    _dtype = None

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
