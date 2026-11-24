"""
Answer Sheet Evaluator - Application Entry Point

Starts the FastAPI server and initializes the evaluation pipeline.

Usage:
    python main.py
    # or
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.evaluation_pipeline import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("Starting %s...", settings.app_name)
    logger.info("Debug mode: %s", settings.debug)

    # Initialize the pipeline (loads TrOCR model)
    pipeline = get_pipeline()
    model_loaded = pipeline.initialize()

    if model_loaded:
        logger.info("Pipeline fully initialized with TrOCR model.")
    else:
        logger.warning(
            "Pipeline initialized WITHOUT TrOCR model. "
            "OCR will not work until the model is available."
        )

    yield

    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(
    title="Answer Sheet Evaluator",
    description=(
        "AI-powered handwritten examination answer-sheet evaluation system. "
        "Supports PDF and image uploads, extracts handwriting via TrOCR, "
        "segments answers by question, evaluates semantically against a rubric, "
        "and produces structured JSON/CSV results with confidence estimation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="", tags=["evaluation"])


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
