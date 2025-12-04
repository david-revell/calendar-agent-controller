"""
Google Calendar MCP Agent (v4)

Professional-grade agent built with the OpenAI Agents SDK.
Uses Phoenix tracing (per-turn spans), an async MCP bridge, and SQLite session
memory to manage real Google Calendar events via list/create/update tools.

Features:
- True think → act (tool) → think loop with final JSON output
- Phoenix tracing for reasoning, tool spans, and MCP result previews
- Per-run session IDs to avoid conversation bleed
- Natural-language date handling (“tomorrow 3pm”, “next Monday”, ISO8601)
- Full integration with the Google Calendar MCP server in this repo
"""

import asyncio
from typing import Optional
from datetime import datetime

from openai import OpenAI  # not strictly needed, but matches qna_agent pattern
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
    with tracer.start_as_current_span("tool_list_calendar_events") as span:
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
    with tracer.start_as_current_span("tool_create_calendar_event") as span:
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
    with tracer.start_as_current_span("tool_update_calendar_event") as span:
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
# Multi-turn REPL entrypoint with session memory
# ---------------------------------------------------------------------

EXIT_COMMANDS = {"exit", "quit", "q"}


# Wrap per-turn reasoning in a chain span for clearer tracing in Phoenix.
@tracer.chain(name="turn_logic")
def run_turn_logic(user_input: str, session: SQLiteSession, turn: int):
    # Attach per-turn context to the chain span so it’s visible without expanding children.
    span = trace.get_current_span()
    if span:
        span.set_attribute("turn", turn)
        span.set_attribute("user_input_preview", user_input[:120])
        span.set_attribute("user_input_len", len(user_input))
    return Runner.run_sync(calendar_agent, user_input, session=session)


# @tracer.agent
@tracer.agent(name="mcp_calender_agent_Attempt4_v3")
def main():
    print("\n=== Google Calendar MCP Agent (Attempt 4 v3) ===")
    print("Type 'exit' to quit.")

    session_id = f"calendar_repl_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    session = SQLiteSession(session_id=session_id, db_path="chat_history.db")
    # Add session context to the root agent span for easier filtering in traces.
    root_span = trace.get_current_span()
    if root_span:
        root_span.set_attribute("session_id", session_id)
    turn = 0

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                print("No input provided.")
                continue

            if user_input.lower() in EXIT_COMMANDS:
                print("Exiting.")
                break

            turn += 1

            with tracer.start_as_current_span(
                "calendar_turn",
                attributes={
                    "session_id": session_id,
                    "turn": turn,
                    "user_input_preview": user_input[:120],
                    "user_input_len": len(user_input),
                },
            ):
                result = run_turn_logic(user_input, session=session, turn=turn)

                # Trace final output for Phoenix (per turn)
                with tracer.start_as_current_span("calendar_agent_final_response") as span:
                    preview = str(result.final_output)[:200] if hasattr(result, "final_output") else ""
                    span.set_attribute("user_input", user_input)
                    span.set_attribute("final_output_preview", preview)
                    span.set_attribute("turn", turn)

            print("\n=== Agent final_output ===")
            print(result.final_output)
    finally:
        close_fn = getattr(session, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
