# Google Calendar MCP Agent (OpenAI Agents SDK)

A lightweight AI agent that reads, creates, and updates Google Calendar
events through an MCP server.\
Built entirely with the **OpenAI Agents SDK**, using **Phoenix
tracing**, **SQLite session memory**, and a **true tool-calling loop**
(`think → act → think → final JSON`).

This project demonstrates real applied AI engineering: an autonomous
agent calling real tools against a live Google Calendar.

## What the agent can do

• List events for any date or range\
• Create meetings from natural language ("book lunch tomorrow at 1pm")\
• Update existing events\
• Detect conflicts\
• Phoenix tracing for every step\
• Per-run session IDs for isolation

## Architecture

    mcp_calendar_agent.py     # The agent
    calendar_mcp_server.py    # MCP server exposing Calendar tools
    tracer_config.py          # Phoenix config
    .gitignore
    requirements.txt

## Running the project

### 1. Virtual environment

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

### 2. Google API credentials

Place `credentials.json` in the project root.

### 3. Run the agent

    python mcp_calendar_agent.py

## MCP Server

Full server documentation moved to:

    Documentation/mcp_server_README.md
