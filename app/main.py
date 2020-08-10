"""Completion rate prediction service.

Environment variables:
- MODEL_DIR: Path to the directory where models are saved. Default `./model`.
- MAX_NUM_MODELS: Maxium number of models sustained at the same time. Default 1.
"""
import logging
import logging.config
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import infer, load

logging_cfg = Path(__file__).parent / "logging.cfg"
logging.config.fileConfig(logging_cfg, disable_existing_loggers=False)
logger = logging.getLogger("serving")

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = str(uuid.uuid4())
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = f"{process_time:.2f}"
    logger.info(
        f"rid={idem} duration={formatted_process_time}ms status_code={response.status_code}"
    )

    return response


@app.get("/")
async def heartbeat():
    """health check."""
    return "ok"


app.include_router(infer.router, prefix="/infer", tags=["infer"])
app.include_router(load.router, prefix="/load", tags=["load"])
