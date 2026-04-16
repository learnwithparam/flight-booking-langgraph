from fastapi import APIRouter, HTTPException
import logging

from models import ChatRequest, ChatResponse, Booking, ServiceInfo
from service import service
from utils.llm_provider import get_provider_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/flight-booking", tags=["flight-booking"])


@router.get("/health", response_model=ServiceInfo)
async def health_check():
    return ServiceInfo(
        status="healthy",
        service="flight-booking-langgraph",
        description=(
            "Multi-turn flight booking agent built as a LangGraph state machine "
            "with checkpointed thread memory and an in-memory flight catalog."
        ),
    )


@router.get("/provider-info")
async def provider_info():
    try:
        config = get_provider_config()
        return {"provider_name": config["provider_name"], "model": config["model"]}
    except Exception as e:
        return {"provider_name": "unknown", "model": "unknown", "error": str(e)}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Run one conversational turn through the booking graph. Pass the same
    thread_id across turns to continue an existing booking flow.
    """
    try:
        result = await service.chat(request.thread_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("chat error")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@router.get("/booking/{booking_id}", response_model=Booking)
async def get_booking_endpoint(booking_id: str):
    row = service.get_booking(booking_id)
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return Booking(**row)
