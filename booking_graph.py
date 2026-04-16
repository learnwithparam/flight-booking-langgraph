"""
Booking graph
=============

A multi-turn flight booking agent modeled as a LangGraph StateGraph. The
conversation flow shapes into small nodes that share a BookingState
TypedDict:

    intent_router
         |
         +--> collect_origin ----+
         |                       |
         +--> collect_destination-+
         |                       |
         +--> collect_dates ------+
         |                       |
         +--> search_flights --> propose_options --> confirm_booking
         |
         +--> smalltalk

A MemorySaver checkpointer persists each thread's state between turns,
keyed on thread_id. Every slot extraction (origin, destination, dates,
passengers, choice) is delegated to the LLM - no regex, no keyword
matching.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils.llm_provider import get_llm_provider
from flight_repo import search_flights, create_booking, list_route_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class BookingState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""
    # Conversation
    messages: List[Dict[str, str]]       # [{role, content}, ...]
    user_message: str
    thread_id: str

    # Slot-filled fields
    origin: Optional[str]
    destination: Optional[str]
    depart_date: Optional[str]
    return_date: Optional[str]
    passengers: Optional[int]

    # Agent outputs
    candidates: List[Dict[str, Any]]
    chosen_flight: Optional[Dict[str, Any]]
    booking_id: Optional[str]

    # Control
    stage: str                           # routing | collecting | searching | proposing | confirming | confirmed | smalltalk
    assistant_reply: str


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

async def _llm_json(prompt: str, system: str) -> Dict[str, Any]:
    """Call the LLM and parse a JSON object out of the response."""
    provider = get_llm_provider()
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
    raw = await provider.generate_text(full_prompt, temperature=0.2, max_tokens=400)
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("LLM JSON parse failed, raw=%r", raw[:200])
        return {}


async def _llm_text(prompt: str, system: str) -> str:
    provider = get_llm_provider()
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
    raw = await provider.generate_text(full_prompt, temperature=0.5, max_tokens=250)
    return (raw or "").strip()


def _append_message(state: BookingState, role: str, content: str) -> None:
    messages = state.get("messages") or []
    messages.append({"role": role, "content": content})
    state["messages"] = messages


# ---------------------------------------------------------------------------
# Node: intent_router
# ---------------------------------------------------------------------------

async def intent_router(state: BookingState) -> BookingState:
    """Classify the turn and extract any slots the user mentioned."""
    system = (
        "You are the router for a flight booking assistant. Classify the user "
        "turn and extract any trip slots present. Return strict JSON with keys: "
        "{\"intent\": one of [book, confirm, smalltalk], "
        "\"origin\": 3-letter IATA code or null, "
        "\"destination\": 3-letter IATA code or null, "
        "\"depart_date\": ISO-ish date string or natural phrase or null, "
        "\"return_date\": date or null, "
        "\"passengers\": integer or null, "
        "\"choice_index\": 1-based integer or null}. "
        "If they respond to a list of options with a choice, set intent=confirm."
    )
    prompt = (
        f"Current slots: origin={state.get('origin')}, destination={state.get('destination')}, "
        f"depart_date={state.get('depart_date')}, return_date={state.get('return_date')}, "
        f"passengers={state.get('passengers')}, stage={state.get('stage')}\n"
        f"User said: {state.get('user_message', '')}\n"
        "Return JSON only."
    )
    data = await _llm_json(prompt, system)

    # Merge any newly extracted slots. Keep prior values if LLM returned null.
    for key in ("origin", "destination", "depart_date", "return_date", "passengers"):
        val = data.get(key)
        if val not in (None, "", "null"):
            state[key] = val  # type: ignore[literal-required]

    # Uppercase IATA codes when they look like codes.
    for key in ("origin", "destination"):
        v = state.get(key)  # type: ignore[literal-required]
        if isinstance(v, str) and len(v) == 3:
            state[key] = v.upper()  # type: ignore[literal-required]

    intent = data.get("intent") or "book"
    state["_intent"] = intent  # type: ignore[typeddict-unknown-key]
    if data.get("choice_index"):
        state["_choice_index"] = int(data["choice_index"])  # type: ignore[typeddict-unknown-key]
    return state


def route_from_intent(state: BookingState) -> str:
    intent = state.get("_intent")  # type: ignore[typeddict-item]
    stage = state.get("stage") or "routing"
    choice = state.get("_choice_index")  # type: ignore[typeddict-item]

    # Already proposed options
    if stage == "proposing":
        if choice or intent == "confirm":
            return "confirm_booking"
        # User is updating slots (passengers, return date) → re-propose
        return "propose_options"
    if intent == "confirm":
        return "confirm_booking"
    if intent == "smalltalk":
        return "smalltalk"

    # Booking flow: fill slots in order
    if not state.get("origin"):
        return "collect_origin"
    if not state.get("destination"):
        return "collect_destination"
    if not state.get("depart_date"):
        return "collect_dates"
    return "search_flights"


# ---------------------------------------------------------------------------
# Slot collection nodes
# ---------------------------------------------------------------------------

async def collect_origin(state: BookingState) -> BookingState:
    pairs = list_route_pairs()
    origins = sorted({p["origin"] for p in pairs})
    state["assistant_reply"] = (
        "Sure, I can help you book a flight. Which city are you flying from? "
        f"We currently serve: {', '.join(origins)}."
    )
    state["stage"] = "collecting"
    return state


async def collect_destination(state: BookingState) -> BookingState:
    origin = state.get("origin") or "your origin"
    pairs = list_route_pairs()
    dests = sorted({p["destination"] for p in pairs if p["origin"] == state.get("origin")})
    hint = f" Available from {origin}: {', '.join(dests)}." if dests else ""
    state["assistant_reply"] = (
        f"Great - flying from {origin}. Where are you headed?{hint}"
    )
    state["stage"] = "collecting"
    return state


async def collect_dates(state: BookingState) -> BookingState:
    origin = state.get("origin")
    dest = state.get("destination")
    state["assistant_reply"] = (
        f"Got it: {origin} to {dest}. What depart date works for you, "
        "and how many passengers? A return date is optional."
    )
    state["stage"] = "collecting"
    return state


# ---------------------------------------------------------------------------
# Node: search_flights
# ---------------------------------------------------------------------------

async def search_flights_node(state: BookingState) -> BookingState:
    origin = state.get("origin") or ""
    dest = state.get("destination") or ""
    depart = state.get("depart_date") or ""

    candidates = search_flights(origin, dest, depart, limit=3)
    state["candidates"] = candidates
    if not candidates:
        state["assistant_reply"] = (
            f"I couldn't find flights from {origin} to {dest}. "
            "Would you like to try a different route?"
        )
        state["stage"] = "collecting"
        # Reset route so the user can retry
        state["origin"] = None
        state["destination"] = None
    return state


# ---------------------------------------------------------------------------
# Node: propose_options
# ---------------------------------------------------------------------------

async def propose_options(state: BookingState) -> BookingState:
    candidates = state.get("candidates") or []
    if not candidates:
        return state

    pax = int(state.get("passengers") or 1)
    lines = []
    for i, c in enumerate(candidates, start=1):
        total = c["price_usd"] * max(1, pax)
        lines.append(
            f"  {i}. {c['carrier']} {c['id']} - depart {c['depart_time']}, "
            f"arrive {c['arrive_time']} ({c['duration_minutes']} min) - "
            f"${c['price_usd']}/pax (total ${total})"
        )
    body = "\n".join(lines)
    depart = state.get("depart_date") or "the requested date"
    state["assistant_reply"] = (
        f"Here are 3 flight options from {state.get('origin')} to {state.get('destination')} "
        f"on {depart} for {pax} passenger(s):\n{body}\n"
        "Reply with the option number (1, 2, or 3) to confirm, or say 'cancel' to start over."
    )
    state["stage"] = "proposing"
    return state


# ---------------------------------------------------------------------------
# Node: confirm_booking
# ---------------------------------------------------------------------------

async def confirm_booking(state: BookingState) -> BookingState:
    options = state.get("candidates") or []
    if not options:
        state["assistant_reply"] = (
            "Let's start over - where would you like to fly from?"
        )
        state["stage"] = "collecting"
        return state

    idx = state.get("_choice_index")  # type: ignore[typeddict-item]
    # If the router didn't extract a choice, ask the LLM explicitly.
    if not isinstance(idx, int):
        system = (
            "You interpret which numbered option the user chose. Return JSON: "
            "{\"choice_index\": 1-based int or null, \"declined\": bool}."
        )
        option_list = "\n".join(
            f"{i+1}. {o['carrier']} {o['id']} at {o['depart_time']}"
            for i, o in enumerate(options)
        )
        prompt = (
            f"Options:\n{option_list}\nUser reply: {state.get('user_message', '')}"
        )
        data = await _llm_json(prompt, system)
        if data.get("declined"):
            state["assistant_reply"] = "No problem - tell me when you'd like to try again."
            state["stage"] = "routing"
            state["candidates"] = []
            return state
        idx = data.get("choice_index")

    if not isinstance(idx, int) or idx < 1 or idx > len(options):
        state["assistant_reply"] = (
            f"Which option works for you? Reply with 1, 2, or {len(options)}."
        )
        state["stage"] = "proposing"
        return state

    chosen = options[idx - 1]
    pax = int(state.get("passengers") or 1)
    booking = create_booking(
        thread_id=state.get("thread_id", "anonymous"),
        flight=chosen,
        depart_date=state.get("depart_date") or "",
        return_date=state.get("return_date"),
        passengers=pax,
    )
    state["chosen_flight"] = chosen
    state["booking_id"] = booking["id"]
    state["stage"] = "confirmed"
    state["assistant_reply"] = (
        f"Booked! Confirmation {booking['id']} - {chosen['carrier']} {chosen['id']} "
        f"from {chosen['origin']} to {chosen['destination']} on "
        f"{state.get('depart_date')} for {pax} passenger(s). "
        f"Total ${booking['total_price_usd']}."
    )
    return state


# ---------------------------------------------------------------------------
# Node: smalltalk
# ---------------------------------------------------------------------------

async def smalltalk(state: BookingState) -> BookingState:
    system = (
        "You are a friendly flight booking assistant. Reply in 1-2 sentences. "
        "If the user seems to want a flight, invite them to share origin and destination."
    )
    prompt = state.get("user_message", "")
    state["assistant_reply"] = await _llm_text(prompt, system)
    state["stage"] = state.get("stage") or "routing"
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(BookingState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("collect_origin", collect_origin)
    graph.add_node("collect_destination", collect_destination)
    graph.add_node("collect_dates", collect_dates)
    graph.add_node("search_flights", search_flights_node)
    graph.add_node("propose_options", propose_options)
    graph.add_node("confirm_booking", confirm_booking)
    graph.add_node("smalltalk", smalltalk)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_from_intent,
        {
            "collect_origin": "collect_origin",
            "collect_destination": "collect_destination",
            "collect_dates": "collect_dates",
            "search_flights": "search_flights",
            "propose_options": "propose_options",
            "confirm_booking": "confirm_booking",
            "smalltalk": "smalltalk",
        },
    )

    graph.add_edge("collect_origin", END)
    graph.add_edge("collect_destination", END)
    graph.add_edge("collect_dates", END)
    graph.add_edge("search_flights", "propose_options")
    graph.add_edge("propose_options", END)
    graph.add_edge("confirm_booking", END)
    graph.add_edge("smalltalk", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED
