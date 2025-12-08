"""
Google Calendar MCP Agent (v3)

Professional-grade agent built with the OpenAI Agents SDK.
Uses Phoenix tracing (per-turn spans), an async MCP bridge, and SQLite session
memory to manage real Google Calendar events via list/create/update tools.

Features:
- True think -> act (tool) -> think loop with final JSON output
- Phoenix tracing for reasoning, tool spans, and MCP result previews
- Per-run session IDs to avoid conversation bleed
- Natural-language date handling ("tomorrow 3pm", "next Monday", ISO8601)
- Full integration with the Google Calendar MCP server in this repo
- LLM-simulated user driver: reads scenarios from CSV, seeds initial prompt, and
  auto-generates subsequent user turns with a max-turn cap (no manual stdin)

Version changes:
- v3: Added plain-text UTF-8 history export with per-turn timestamps and ensured the final user message is logged exactly once. Minor tracing-decorator cleanup.
- v2: Introduced LLM-driven synthetic user simulator, scenario CSV loading, and max-turn limit (replacing manual stdin loop).
- v1: Baseline REPL-driven agent with MCP calendar tools and tracing.


Change request (v2.1 -> v3):
"Apply the following minimal fixes to the existing file without changing the overall structure or adding new abstractions:
1. Fix final-turn logging
Ensure the last user message is logged once and only if the agent did not process it. Remove any duplicate logging caused by the finally block.
2. Clean up the history writer
Keep the history file in the same human-readable plain-text format as the example transcript (user turns as free text, assistant turns as their JSON final_output). Ensure timestamps appear once per turn and in correct order.
3. Standardise tracing decorators
Ensure each MCP tool has a single @tracer.tool decorator and the main turn loop has a single @tracer.agent or @tracer.chain decorator. Remove any duplicate or conflicting decorators but do not change the tracing structure beyond this."

"""

import asyncio
import csv  # to load the scenarios CSV
import json  # the simulator LLM returns JSON ({"message":..., "continue":...}).
import os  # to check file existence and read SCENARIO_NAME env var.
from typing import Optional, Tuple, List, Dict  # typing for simulate_user_turn, load_scenarios, etc.
from datetime import datetime

from openai import OpenAI
from agents import Agent, Runner, function_tool
from agents.memory.sqlite_session import SQLiteSession
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from tracer_config import tracer
from opentelemetry import trace

client = OpenAI()

# ---------------------------------------------------------------------
# MCP bridge (async)
# ---------------------------------------------------------------------

SERVER_COMMAND = "python"
SERVER_ARGS = ["calendar_mcp_server.py"]  # same dir as this file


async def _call_mcp(tool_name: str, args: dict) -> str:
    """Low-level MCP call, returns plain text for the agent."""
    params = StdioServerParameters(
        command=SERVER_COMMAND,
        args=SERVER_ARGS,
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)

    # Try to collapse the MCP response into a simple string
    try:
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                parts = []
                for c in content:
                    # Text blocks usually have .text or are simple strings
                    text = getattr(c, "text", None)
                    parts.append(text if text is not None else str(c))
                return "\n".join(parts)
            return str(content)
        return str(result)
    except Exception:
        return str(result)


async def call_mcp(tool_name: str, args: dict) -> str:
    """Async wrapper so tools can await MCP without blocking the event loop."""
    return await _call_mcp(tool_name, args)


# ---------------------------------------------------------------------
# Tools exposed to the Agent (SDK @function_tool)
# ---------------------------------------------------------------------
@function_tool
async def list_calendar_events(date_start: str, date_end: Optional[str] = None) -> str:
    """
    List calendar events between date_start and date_end (inclusive).
    Dates can be 'today', 'tomorrow', 'next Monday', '2025-11-26', etc.
    """
    with tracer.start_as_current_span("list_calendar_events") as span:
        span.set_attribute("date_start", date_start)
        if date_end:
            span.set_attribute("date_end", date_end)

        try:
            text = await call_mcp(
                "list_events",
                {
                    "date_start": date_start,
                    "date_end": date_end,
                },
            )
            span.set_attribute("mcp_result_preview", text[:200])
            return text
        except Exception as exc:
            span.add_event("tool_exception", {"tool": "list_events", "error": str(exc)})
            raise


@function_tool
async def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[str] = None,
) -> str:
    """
    Create a new calendar event via MCP.
    Datetimes can be ISO8601 or natural language like 'tomorrow 3pm'.
    Attendees is an optional comma-separated list of emails.
    """
    with tracer.start_as_current_span("create_calendar_event") as span:
        span.set_attribute("summary", summary)
        span.set_attribute("start_datetime", start_datetime)
        span.set_attribute("end_datetime", end_datetime)

        args = {
            "summary": summary,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": description,
            "location": location,
            "attendees": attendees,
        }

        try:
            text = await call_mcp("create_event", args)
            span.set_attribute("mcp_result_preview", text[:200])
            return text
        except Exception as exc:
            span.add_event("tool_exception", {"tool": "create_event", "error": str(exc)})
            raise


@function_tool
async def update_calendar_event(
    event_id: str,
    summary: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """
    Update an existing calendar event by Google Calendar event_id.
    Any field left as None will be left unchanged.
    """
    with tracer.start_as_current_span("update_calendar_event") as span:
        span.set_attribute("event_id", event_id)
        if summary:
            span.set_attribute("summary", summary)
        if start_datetime:
            span.set_attribute("start_datetime", start_datetime)
        if end_datetime:
            span.set_attribute("end_datetime", end_datetime)

        args = {
            "event_id": event_id,
            "summary": summary,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": description,
            "location": location,
        }

        try:
            text = await call_mcp("update_event", args)
            span.set_attribute("mcp_result_preview", text[:200])
            return text
        except Exception as exc:
            span.add_event("tool_exception", {"tool": "update_event", "error": str(exc)})
            raise


# ---------------------------------------------------------------------
# Agent definition (true SDK agent, with tools + JSON final output)
# ---------------------------------------------------------------------

calendar_agent = Agent(
    name="GoogleCalendarAgent",
    model="gpt-5-nano",
    instructions="""
You are an autonomous medical receptionist assistant for David's Google Calendar.

You have three tools:
- list_calendar_events(date_start, date_end?)
- create_calendar_event(summary, start_datetime, end_datetime, description?, location?, attendees?)
- update_calendar_event(event_id, summary?, start_datetime?, end_datetime?, description?, location?)

General behaviour:
- Think step-by-step: PLAN -> use tools -> observe results -> PLAN again until the task is complete.
- You may call tools multiple times in one run.
- Use natural language time like "today", "tomorrow", "next Thursday" when helpful; the MCP server can parse them.

Guidelines:
- When the user wants to SEE events, call list_calendar_events with an appropriate range.
- When the user wants to CREATE an event:
  - If any essential detail is missing (title, start, end), ask the user to clarify before calling the tool.
  - If the user has not provided any notes or symptoms for a medical appointment, ask ONCE: "Would you like to add any notes or symptoms to this appointment?"
  - If the user has not provided a location, you may ask ONCE: "Would you like to specify a location for this appointment?" Do not suggest example location types.
  - When asking for missing duration or time details, do not propose example options (e.g., do not suggest 30 or 60 minutes); request the info neutrally.
  - Assume the user's local timezone for all times; do NOT ask which timezone to use.
  - Before creating, check for conflicts using list_calendar_events for the relevant time/day. Do NOT double-book. If there is a conflict, tell the user it conflicts and propose an alternative nearby free time, then ask for confirmation or another time.
  - Otherwise call create_calendar_event.
- When the user wants to UPDATE an event:
  - If the event_id is unknown, first call list_calendar_events (for the relevant day or range),
    show options, and ask the user which Event ID to use.
  - Then call update_calendar_event with the chosen event_id and new fields.

Final answer format (always):
- When you are finished (no more tool calls needed), respond with VALID JSON only, no extra text.
- The JSON MUST have exactly these keys:
  {
    "final_answer": "<natural language summary to the user>",
    "reasoning": "<short explanation of the tools used and why>"
  }
- final_answer should be one or two short sentences, suitable to show directly to the user.
- reasoning can mention which tools were called and any important decisions.
""",
    tools=[list_calendar_events, create_calendar_event, update_calendar_event],
)

# ---------------------------------------------------------------------
# Multi-turn loop now driven by an LLM user simulator (no manual stdin)
# ---------------------------------------------------------------------

SCENARIOS_CSV = "scenarios.csv"
DEFAULT_MAX_TURNS = 10
# Model used for the synthetic user simulator; overridable via SIMULATED_USER_MODEL env var.
SIMULATED_USER_MODEL = os.getenv("SIMULATED_USER_MODEL", "gpt-5-nano")


def load_scenarios(csv_path: str) -> List[Dict[str, str]]:
    """Load scenario rows from CSV so we can drive synthetic conversations."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("scenario") and row.get("initial_user_message")]


def format_history(history: List[Dict[str, str]]) -> str:
    """Render a compact text history for the simulator prompt."""
    lines = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def simulate_user_turn(
    scenario: str,
    last_agent_reply: str,
    conversation_history: List[Dict[str, str]],
) -> Tuple[str, bool]:
    """
    Ask the LLM to produce the next synthetic user message.
    Returns (message, continue_flag). The continue_flag lets us stop early.
    """
    # Keep the simulator focused on the scenario and short outputs.
    system_prompt = (
        "You simulate the user in a conversation with a calendar agent. "
        "Stay consistent with the scenario. If the task is done or blocked, set continue to false."
    )

    history_text = format_history(conversation_history)
    user_prompt = (
        "Respond with JSON: {\"message\": \"<next user message>\", \"continue\": true|false}.\n"
        f"Scenario: {scenario}\n"
        f"Agent reply: {last_agent_reply}\n"
        f"Conversation so far:\n{history_text}"
    )

    # Some models (e.g., gpt-5-nano) reject non-default temperature; omit it to stay compatible.
    completion = client.chat.completions.create(
        model=SIMULATED_USER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = completion.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("message", "").strip(), bool(parsed.get("continue", True))
    except Exception:
        # If parsing fails, treat the raw text as the message and continue.
        return raw, True


# Wrap per-turn reasoning in a chain span for clearer tracing in Phoenix.
@tracer.agent(name="turn_logic")
def run_turn_logic(user_input: str, session: SQLiteSession, turn: int):
    # Attach per-turn context to the chain span so it is visible without expanding children.
    span = trace.get_current_span()
    if span:
        span.set_attribute("turn", turn)
        span.set_attribute("user_input_preview", user_input[:120])
        span.set_attribute("user_input_len", len(user_input))
    return Runner.run_sync(calendar_agent, user_input, session=session)


def choose_scenario(scenarios: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Pick a scenario by environment variable SCENARIO_NAME, otherwise first row.
    This keeps selection deterministic for automated runs.
    """
    if not scenarios:
        return None
    desired = os.getenv("SCENARIO_NAME")
    if not desired:
        return scenarios[0]
    for row in scenarios:
        if row.get("scenario") == desired:
            return row
    return scenarios[0]


def save_history(scenario:str,conversation_history: List[Dict[str, str]]):
    """Save the conversation history to a file."""
    history_string = f"Scenario: {scenario}\nHistory:\n\n"
    for turn in conversation_history:
        history_string += f" - {turn['role']} [{turn['timestamp']}]:\n {turn['content']}\n"

    # Write as UTF-8 so smart quotes/emoji don't get mangled in transcripts.
    with open(f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
        f.write(history_string)

# @tracer.agent
@tracer.agent(name="mcp_calender_agent")
def main():
    print("\n=== Google Calendar MCP Agent (LLM-simulated user) ===")

    scenarios = load_scenarios(SCENARIOS_CSV)
    scenario_row = choose_scenario(scenarios)
    if not scenario_row:
        print(f"No scenarios found in {SCENARIOS_CSV}. Add rows with scenario and initial_user_message.")
        return

    scenario = scenario_row["scenario"]
    user_input = scenario_row["initial_user_message"]
    max_turns = int(os.getenv("MAX_TURNS", DEFAULT_MAX_TURNS))

    session_id = f"calendar_repl_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    session = SQLiteSession(session_id=session_id, db_path="chat_history.db")
    # Add session context to the root agent span for easier filtering in traces.
    root_span = trace.get_current_span()
    if root_span:
        root_span.set_attribute("session_id", session_id)
        # Record which models are in play for downstream observability.
        root_span.set_attribute("calendar_agent_model", calendar_agent.model)
        root_span.set_attribute("controller_model", SIMULATED_USER_MODEL)
        root_span.set_attribute("scenario", scenario)
    turn = 0
    conversation_history: List[Dict[str, str]] = []
    continue_flag = True
    unprocessed_final_user: Optional[str] = None

    try:
        while continue_flag and turn < max_turns:
            turn += 1

            with tracer.start_as_current_span(
                "calendar_turn",
                attributes={
                    "session_id": session_id,
                    "turn": turn,
                    "user_input_preview": user_input[:120],
                    "user_input_len": len(user_input),
                    "scenario": scenario,
                },
            ):
                result = run_turn_logic(user_input, session=session, turn=turn)
                agent_reply = str(result.final_output) if hasattr(result, "final_output") else str(result)

                # Trace final output for Phoenix (per turn)
                with tracer.start_as_current_span("calendar_agent_final_response") as span:
                    preview = agent_reply[:200]
                    span.set_attribute("user_input", user_input)
                    span.set_attribute("final_output_preview", preview)
                    span.set_attribute("turn", turn)
                    span.set_attribute("scenario", scenario)

            print(f"\n=== Agent final_output (turn {turn}) ===")
            print(agent_reply)

            # Track conversation for the simulator prompt.
            conversation_history.append(
                {
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": agent_reply,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            if turn >= max_turns:
                print(f"Reached max turns ({max_turns}); stopping.")
                break

            # Ask the LLM to produce the next synthetic user turn.
            user_input, continue_flag = simulate_user_turn(
                scenario=scenario,
                last_agent_reply=agent_reply,
                conversation_history=conversation_history,
            )
            if not continue_flag:
                unprocessed_final_user = user_input

    finally:
        close_fn = getattr(session, "close", None)
        if unprocessed_final_user:
            conversation_history.append(
                {
                    "role": "user",
                    "content": unprocessed_final_user,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        save_history(scenario, conversation_history)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
