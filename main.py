from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from contextlib import asynccontextmanager
from typing import List, Optional
from pymilvus import connections
import gc
import os
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model and preprocessing once on startup

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"All models will run on {device}")

    load_dotenv(override=True)

    uri = os.environ.get("MILVUS_URI")
    token = os.environ.get("MILVUS_TOKEN")
    user = os.environ.get("MILVUS_USER", "User")
    password = os.environ.get("MILVUS_PASSWORD", "Password")

    try:
        connections.connect(
            "default",
            user=user,
            password=password,
            uri=uri,
            token=token
        )
    except Exception as e:
        print("Retry to find if in local has Milvus")
        try:
            connections.connect("default", host="localhost", port="19530")
        except Exception as e:
            print("Failed to connect to database. Stopping the process.")
            raise e


    # Initialize libraries (singleton models and preprocessors)
    from libs.search_by_scenes import init as init_scenes
    from libs.search_by_asr import init as init_asr
    from libs.search_by_keyframes import init as init_keyframes

    embedding_llama_server_url = os.environ.get("EMBEDDING_LLAMA_SERVER_URL", "http://localhost:8080/embedding")

    await init_scenes(device)
    await init_asr(embedding_llama_server_url)
    await init_keyframes(device)
    
    yield

    try:
        from libs.search_by_scenes import shutdown as shutdown_scenes
        from libs.search_by_asr import shutdown as shutdown_asr
        from libs.search_by_keyframes import shutdown as shutdown_keyframes

        await shutdown_scenes()
        await shutdown_asr()
        await shutdown_keyframes()
    finally:
        connections.disconnect("default")
        
        # Clear GPU cache if used
        if device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()

app = FastAPI(lifespan=lifespan)

app.title = "Waiedu FastAPI Server"
app.description = "API for Waiedu video search and retrieval system"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


from libs.search_by_scenes import search_by_scenes

class SceneSearchRequest(BaseModel):
    query: str
    top_k: int = Field(..., ge=16, description="Number of top results, minimum 16")

@app.post("/scene_search")
async def scene_search(request: SceneSearchRequest, subset: Optional[str] = Query(None, description="Comma-separated list of ID prefixes to filter results")):
    # Parse subset parameter if provided
    subset_list = None
    if subset:
        subset_list = [s.strip() for s in subset.split(',') if s.strip()]
    
    try:
        results = search_by_scenes(request.query, request.top_k, subset_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    keyframe_ids = [kf for kf, _ in results]
    return {"keyframes": keyframe_ids}
    
from libs.search_by_asr import search_by_asr

class AsrSearchRequest(BaseModel):
    query: str
    top_k: int = Field(..., ge=16, description="Number of top results, minimum 16")

@app.post("/asr_search")
async def asr_search(request: AsrSearchRequest, subset: Optional[str] = Query(None, description="Comma-separated list of ID prefixes to filter results")):
    # Parse subset parameter if provided
    subset_list = None
    if subset:
        subset_list = [s.strip() for s in subset.split(',') if s.strip()]
    
    try:
        results = search_by_asr(request.query, request.top_k, subset_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    keyframe_ids = [kf for kf, _ in results]
    return {"keyframes": keyframe_ids}

from libs.search_by_keyframes import search_by_keyframes

class KeyframeSearchRequest(BaseModel):
    image_id_query: Optional[str] = Field(None, description="ID of the image to search for")
    text_query: Optional[str] = Field(None, description="Text query to search for")
    top_k: int = Field(..., ge=16, description="Number of top results, minimum 16")

@app.post("/keyframe_search")
async def keyframe_search(request: KeyframeSearchRequest, subset: Optional[str] = Query(None, description="Comma-separated list of ID prefixes to filter results")):
    if not request.image_id_query and not request.text_query:
        raise HTTPException(status_code=400, detail="Either image ID or text query must be provided.")

    # Parse subset parameter if provided
    subset_list = None
    if subset:
        subset_list = [s.strip() for s in subset.split(',') if s.strip()]

    try:
        results = search_by_keyframes(request.image_id_query, request.text_query, request.top_k, subset_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    keyframe_ids = [kf for kf, _ in results]
    return {"keyframes": keyframe_ids}

# Bulk search endpoint
class BulkSearchQuery(BaseModel):
    type: str
    args: dict

@app.post("/search")
async def bulk_search(queries: List[BulkSearchQuery], subset: Optional[str] = Query(None, description="Comma-separated list of ID prefixes to filter results")):
    rrf_buckets: dict[str, list[int]] = {}
    fused_list: list[tuple[str, float]] = []

    for query in queries:
        qtype = query.type.lower()
        args = query.args
        if qtype not in ['asr', 'scenes', 'keyframes']:
            raise HTTPException(status_code=400, detail=f"Unsupported search type: {qtype}")

        # Parse subset parameter if provided
        subset_list = None
        if subset:
            subset_list = [s.strip() for s in subset.split(',') if s.strip()]

        try:
            if qtype == 'asr':
                req = AsrSearchRequest(**args)
                res = search_by_asr(req.query, req.top_k, subset_list)
            elif qtype == 'scenes':
                req = SceneSearchRequest(**args)
                res = search_by_scenes(req.query, req.top_k, subset_list)
            elif qtype == 'keyframes':
                req = KeyframeSearchRequest(**args)
                if not req.image_id_query and not req.text_query:
                    raise HTTPException(status_code=400, detail="Either image ID or text query must be provided.")

                res = search_by_keyframes(req.image_id_query, req.text_query, req.top_k, subset_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # collect keyframe+distance pairs
        last_distance = None
        last_rank = 0
        for _, (keyframe, distance) in enumerate(res):
            if distance != last_distance:
                rank = last_rank + 1
            else:
                rank = last_rank
            last_distance = distance
            last_rank = rank
            rrf_buckets.setdefault(keyframe, []).append(rank)

    RRF_K = 60

    # compute and sort by fused RRF score
    for keyframe, ranks in rrf_buckets.items():
        score = sum(1.0 / (RRF_K + r) for r in ranks)
        fused_list.append((keyframe, score))
    fused_list.sort(key=lambda x: x[1], reverse=True)

    return {
        "keyframes": [kf for kf, _ in fused_list],
        "scores": [score for _, score in fused_list]
    }

if __name__ == "__main__":
    PORT = 8250

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)