import logging
from starlette import status
from starlette.responses import JSONResponse
from fastapi import APIRouter, BackgroundTasks
from app.pipelines.pipeline_trigger import run_indexing_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Pipeline Trigger"],
)


@router.post("/trigger-indexing", status_code=status.HTTP_202_ACCEPTED)
async def trigger_indexing(background_tasks: BackgroundTasks):
    logger.info("Received request to trigger indexing pipeline.")
    
    background_tasks.add_task(run_indexing_pipeline)
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": "Indexing job accepted. The process is running in the background. Check server logs for progress."}
    )