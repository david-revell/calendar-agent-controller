# Copilot Instructions for google-calendar-mcp

## Project Overview
- This project is a Model Context Protocol (MCP) server for Google Calendar, enabling LLM agents (e.g., Claude, Copilot, Cursor) to access and manage calendar events via a standardized interface.
- Key files: `calendar_mcp_server.py` (main server), `mcp_calendar_agent_v*.py` (agent definitions/versions), `requirements.txt` (dependencies), `credentials.json` (Google OAuth), `README.md` (usage/workflows).

## Architecture & Data Flow
- The server exposes Google Calendar as MCP resources and tools, supporting event listing, creation, and updates.
- Agents are defined in `mcp_calendar_agent_v*.py` and interact with the server via MCP transports (typically stdio or HTTP).
- Authentication uses OAuth2; credentials are stored in `token.json` after first login.
- Prompts and tools are surfaced to clients for structured, guided interactions.

## Developer Workflows
- **Setup:**
  - Requires Python 3.10+ and a valid `credentials.json` from Google Cloud Console.
  - Install dependencies: `pip install -r requirements.txt` or `uv pip install -r requirements.txt` (recommended).
- **Run/Debug:**
  - Start server: `python calendar_mcp_server.py` or `mcp dev calendar_mcp_server.py` (for Inspector/debugging).
  - Use with Claude Desktop, Cursor, or other MCP clients by configuring the server command as described in `README.md`.
- **Testing:**
  - Test scripts: `agent_test.py`, `mcp_calendar_test.py`.
  - Use MCP Inspector for interactive debugging and protocol validation.

## Project-Specific Conventions
- All agent versions are in `mcp_calendar_agent_v*.py` (v1–v6, plus main and FAIL variants). Each version may use different prompt/response strategies or model settings.
- The main server entry point is always `calendar_mcp_server.py` (or versioned variants for experiments).
- Prompts and tools are explicitly defined and surfaced for client discovery (see `README.md` and code comments).
- Use UTC for all timestamps; local time is only for display.
- Store sensitive credentials in `credentials.json` and `token.json` (never commit these files).

## Integration & Dependencies
- Integrates with Google Calendar API via `google-api-python-client`, `google-auth`, and related libraries.
- Uses the `mcp` Python package for protocol support.
- Optional: `openai`, `llama-index`, and tracing/telemetry packages for advanced features.

## Examples
- List events: `calendar://events/today` or use the `list_events` tool.
- Create event: Use the `create_event` tool or prompt (e.g., "Schedule a meeting at 2pm tomorrow").
- Update event: Use the `update_event` tool with the event ID.

## References
- See `README.md` for full setup, usage, and troubleshooting.
- See `Documentation/mcp_llm.md` for MCP protocol details, agent patterns, and advanced workflows.

---
**For AI agents:**
- Always check for the latest agent/server version before making changes.
- Follow the explicit prompt and tool definitions in the codebase.
- When adding new features, update both the server and agent files, and document changes in `results/mcp_calendar_agent_v4_changelog.md` if relevant.
