from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """Incoming chat turn from the traveler."""
    message: str = Field(..., description="User utterance for this turn")
    thread_id: str = Field(..., description="Conversation thread id. Reuse to continue a booking.")


class ChatResponse(BaseModel):
    """Response from one turn of the booking graph."""
    thread_id: str
    reply: str
    stage: str
    state: Dict[str, Any]


class FlightOption(BaseModel):
    """A single flight offer returned to the traveler."""
    id: str
    carrier: str
    origin: str
    destination: str
    depart_date: str
    depart_time: str
    arrive_time: str
    duration_minutes: int
    price_usd: int


class Booking(BaseModel):
    """A confirmed booking record."""
    id: str
    thread_id: str
    flight_id: str
    carrier: str
    origin: str
    destination: str
    depart_date: str
    return_date: Optional[str] = None
    passengers: int
    total_price_usd: int
    status: str
    created_at: str


class ServiceInfo(BaseModel):
    """Health / identity response."""
    status: str
    service: str
    description: str
