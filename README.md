# Flight Booking Agent with LangGraph

![learnwithparam.com](https://www.learnwithparam.com/ai-bootcamp/opengraph-image)

Build a multi-turn flight booking agent as a LangGraph state machine. Route intents, collect trip slots across turns, propose flight options, and confirm bookings, all through one FastAPI endpoint with checkpointed thread memory.

> Start learning at [learnwithparam.com](https://learnwithparam.com). Regional pricing available with discounts of up to 60%.

## What You'll Learn

- Model a multi-turn booking conversation as a LangGraph `StateGraph`
- Share a single `BookingState` TypedDict across intent routing, slot collection, search, and confirmation nodes
- Persist per-conversation state with `MemorySaver` keyed on `thread_id`
- Delegate every slot extraction (origin, destination, dates, passengers, choice) to the LLM, no regex
- Compose an in-memory flight catalog that stands in for a real GDS integration
- Wire the whole flow into a single `POST /chat` endpoint with a clean JSON response

## Tech Stack

- **FastAPI** - High-performance async Python web framework
- **LangGraph** - State machine orchestration with checkpointed thread memory
- **OpenRouter** - Default LLM provider (Fireworks, Gemini, OpenAI also supported)
- **Pydantic** - Request and response validation
- **LLM Provider Pattern** - Swap providers without touching graph code
- **Docker** - Containerized development

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (installed automatically by `make setup`)
- An API key from any supported LLM provider

### Quick Start

```bash
# One command to set up and run
make dev

# Or step by step:
make setup          # Create .env and install dependencies
# Edit .env with your API key
make run            # Start the FastAPI server
```

### With Docker

```bash
make build
make up
make logs
make down
```

### API Documentation

Open [http://localhost:8000/docs](http://localhost:8000/docs) once the server is running.

Primary endpoints:

- `POST /flight-booking/chat` - send one turn, receive the agent reply plus state snapshot
- `GET /flight-booking/booking/{booking_id}` - fetch a confirmed booking
- `GET /flight-booking/health` - service health check

## Example Conversation

```
User:  I want to fly from LHR to JFK next Friday
Agent: Got it: LHR to JFK. What depart date works for you, and how many passengers?

User:  2 passengers, return next Sunday
Agent: Here are 3 flight options from LHR to JFK on next Friday for 2 passenger(s):
         1. British Airways FL-BA117 - depart 09:30, arrive 12:45 - $612/pax (total $1224)
         2. Virgin Atlantic FL-VS003 - depart 14:10, arrive 17:25 - $578/pax (total $1156)
         3. Delta FL-DL401 - depart 18:55, arrive 22:05 - $545/pax (total $1090)
       Reply with the option number to confirm.

User:  2
Agent: Booked! Confirmation BK-7A3F9C21 - Virgin Atlantic FL-VS003 from LHR to JFK
       on next Friday for 2 passenger(s). Total $1156.
```

## Challenges

Work through these incrementally to build the full application:

1. **Shape the shared state** - Define `BookingState` with `messages`, `origin`, `destination`, `depart_date`, `return_date`, `passengers`, `candidates`, `chosen_flight`, `booking_id`, `stage`
2. **Route intents with the LLM** - Classify each turn into `book`, `confirm`, or `smalltalk` and extract any slots present in the same call
3. **Collect missing slots across turns** - Add `collect_origin`, `collect_destination`, `collect_dates` nodes that ask targeted follow-ups
4. **Search the flight catalog** - Seed 15 flights across 5 routes and query by origin, destination, depart date
5. **Propose options and confirm** - Show 3 candidates, have the LLM interpret the user's choice, and write a booking record
6. **Persist with MemorySaver** - Wire `MemorySaver()` into `graph.compile(checkpointer=...)` so every turn resumes from the last checkpoint keyed on `thread_id`

## Makefile Targets

```
make help           Show all available commands
make setup          Initial setup (create .env, install deps)
make dev            Setup and run (one command!)
make run            Start FastAPI server
make build          Build Docker image
make up             Start container
make down           Stop container
make clean          Remove venv and cache
```

## About

Created by [learnwithparam.com](https://learnwithparam.com) - Learn AI Engineering through hands-on projects.

Explore more courses at [learnwithparam.com/courses](https://learnwithparam.com/courses).
