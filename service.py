"""
Service layer between the FastAPI router and the LangGraph booking graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from booking_graph import get_graph
from flight_repo import get_booking

logger = logging.getLogger(__name__)


class FlightBookingService:
    """Thin wrapper around the compiled LangGraph."""

    def __init__(self, graph, repo_get_booking):
        self._graph = graph
        self._get_booking = repo_get_booking

    async def chat(self, thread_id: str, user_message: str) -> Dict[str, Any]:
        """Run one turn through the graph. MemorySaver resumes prior state."""
        config = {"configurable": {"thread_id": thread_id}}

        # Seed this turn's input. The checkpointer merges with prior state.
        turn_input: Dict[str, Any] = {
            "user_message": user_message,
            "thread_id": thread_id,
        }
        # Append the user message to running transcript.
        prior = self._graph.get_state(config)
        prior_messages = []
        if prior and prior.values:
            prior_messages = list(prior.values.get("messages") or [])
        prior_messages.append({"role": "user", "content": user_message})
        turn_input["messages"] = prior_messages

        result = await self._graph.ainvoke(turn_input, config=config)

        reply = result.get("assistant_reply", "")
        if reply:
            result_messages = list(result.get("messages") or [])
            result_messages.append({"role": "assistant", "content": reply})
            # Persist the assistant message back into the checkpoint.
            self._graph.update_state(config, {"messages": result_messages})

        # Build a JSON-safe snapshot of state (drop private keys).
        snapshot = {
            k: v for k, v in result.items()
            if not k.startswith("_")
        }

        return {
            "thread_id": thread_id,
            "reply": reply,
            "stage": result.get("stage") or "routing",
            "state": snapshot,
        }

    def get_booking(self, booking_id: str) -> Optional[Dict[str, Any]]:
        return self._get_booking(booking_id)


# Singleton used by the router
service = FlightBookingService(get_graph(), get_booking)
