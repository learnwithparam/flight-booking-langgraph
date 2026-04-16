"""
In-memory seed catalog for flights and bookings.

Fifteen flights across five origin-destination pairs. Real systems would
hit a GDS or airline API here, but an in-memory repo keeps the workshop
focused on the LangGraph flow.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_FLIGHTS: List[Dict[str, Any]] = [
    # LHR -> JFK
    {"id": "FL-BA117", "carrier": "British Airways", "origin": "LHR", "destination": "JFK",
     "depart_time": "09:30", "arrive_time": "12:45", "duration_minutes": 495, "price_usd": 612},
    {"id": "FL-VS003", "carrier": "Virgin Atlantic", "origin": "LHR", "destination": "JFK",
     "depart_time": "14:10", "arrive_time": "17:25", "duration_minutes": 495, "price_usd": 578},
    {"id": "FL-DL401", "carrier": "Delta", "origin": "LHR", "destination": "JFK",
     "depart_time": "18:55", "arrive_time": "22:05", "duration_minutes": 490, "price_usd": 545},

    # SFO -> NRT
    {"id": "FL-UA837", "carrier": "United", "origin": "SFO", "destination": "NRT",
     "depart_time": "11:15", "arrive_time": "15:40+1", "duration_minutes": 665, "price_usd": 895},
    {"id": "FL-JL001", "carrier": "Japan Airlines", "origin": "SFO", "destination": "NRT",
     "depart_time": "13:45", "arrive_time": "18:10+1", "duration_minutes": 665, "price_usd": 942},
    {"id": "FL-NH007", "carrier": "ANA", "origin": "SFO", "destination": "NRT",
     "depart_time": "22:30", "arrive_time": "03:05+2", "duration_minutes": 635, "price_usd": 1025},

    # DEL -> DXB
    {"id": "FL-EK511", "carrier": "Emirates", "origin": "DEL", "destination": "DXB",
     "depart_time": "04:20", "arrive_time": "06:35", "duration_minutes": 215, "price_usd": 312},
    {"id": "FL-AI915", "carrier": "Air India", "origin": "DEL", "destination": "DXB",
     "depart_time": "10:50", "arrive_time": "13:05", "duration_minutes": 215, "price_usd": 285},
    {"id": "FL-6E1401", "carrier": "IndiGo", "origin": "DEL", "destination": "DXB",
     "depart_time": "20:15", "arrive_time": "22:30", "duration_minutes": 215, "price_usd": 248},

    # LAX -> SYD
    {"id": "FL-QF012", "carrier": "Qantas", "origin": "LAX", "destination": "SYD",
     "depart_time": "22:25", "arrive_time": "06:30+2", "duration_minutes": 905, "price_usd": 1285},
    {"id": "FL-UA863", "carrier": "United", "origin": "LAX", "destination": "SYD",
     "depart_time": "21:00", "arrive_time": "05:25+2", "duration_minutes": 925, "price_usd": 1198},
    {"id": "FL-DL041", "carrier": "Delta", "origin": "LAX", "destination": "SYD",
     "depart_time": "23:10", "arrive_time": "07:15+2", "duration_minutes": 905, "price_usd": 1242},

    # CDG -> FRA
    {"id": "FL-AF1118", "carrier": "Air France", "origin": "CDG", "destination": "FRA",
     "depart_time": "07:25", "arrive_time": "08:50", "duration_minutes": 85, "price_usd": 142},
    {"id": "FL-LH1051", "carrier": "Lufthansa", "origin": "CDG", "destination": "FRA",
     "depart_time": "12:40", "arrive_time": "14:05", "duration_minutes": 85, "price_usd": 158},
    {"id": "FL-LH1053", "carrier": "Lufthansa", "origin": "CDG", "destination": "FRA",
     "depart_time": "18:15", "arrive_time": "19:40", "duration_minutes": 85, "price_usd": 169},
]


_BOOKINGS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Flight search
# ---------------------------------------------------------------------------

def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Return up to `limit` flight offers stamped with the requested depart date."""
    origin_u = (origin or "").upper().strip()
    dest_u = (destination or "").upper().strip()
    matches: List[Dict[str, Any]] = []
    for f in _FLIGHTS:
        if f["origin"] == origin_u and f["destination"] == dest_u:
            offer = dict(f)
            offer["depart_date"] = depart_date
            matches.append(offer)
        if len(matches) >= limit:
            break
    return matches


def get_flight(flight_id: str) -> Optional[Dict[str, Any]]:
    for f in _FLIGHTS:
        if f["id"] == flight_id:
            return dict(f)
    return None


def list_route_pairs() -> List[Dict[str, str]]:
    """Unique origin/destination pairs. Handy for the user when they ask what's bookable."""
    seen: Dict[str, Dict[str, str]] = {}
    for f in _FLIGHTS:
        key = f"{f['origin']}-{f['destination']}"
        if key not in seen:
            seen[key] = {"origin": f["origin"], "destination": f["destination"]}
    return list(seen.values())


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------

def create_booking(
    thread_id: str,
    flight: Dict[str, Any],
    depart_date: str,
    return_date: Optional[str],
    passengers: int,
) -> Dict[str, Any]:
    booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"
    total = int(flight["price_usd"]) * max(1, int(passengers or 1))
    row = {
        "id": booking_id,
        "thread_id": thread_id,
        "flight_id": flight["id"],
        "carrier": flight["carrier"],
        "origin": flight["origin"],
        "destination": flight["destination"],
        "depart_date": depart_date,
        "return_date": return_date,
        "passengers": int(passengers or 1),
        "total_price_usd": total,
        "status": "confirmed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _BOOKINGS[booking_id] = row
    return row


def get_booking(booking_id: str) -> Optional[Dict[str, Any]]:
    return _BOOKINGS.get(booking_id)
