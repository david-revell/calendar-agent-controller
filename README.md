# Calendar Agent Evaluation Framework

Evaluation harness for testing **MCP-based AI agents** using synthetic user scenarios, conversation logs, and Phoenix tracing.

The framework is used to run repeatable, multi-turn conversations against a tool-calling agent and inspect how it behaves under realistic and adversarial conditions.

---

## Scope

This repository focuses on **agent evaluation**, not agent features.

It provides:
- a multi-turn execution loop with termination handling  
- a scenario-driven synthetic user  
- structured transcripts and tracing for inspection  

It does not provide:
- a user-facing application  
- a production deployment  
- prompt-only experiments  

---

## Evaluation approach

Conversations are driven by CSV-defined scenarios and an LLM-based synthetic user.  
Each run executes until completion, failure, or a max-turn limit is reached.

The output of each run includes:
- a plain-text transcript with timestamps  
- an explicit stop reason when termination occurs  
- Phoenix traces showing per-turn reasoning and tool spans  

These artefacts are used for inspection and debugging rather than scoring.

---

## Observed failure cases

The following behaviours have been observed during evaluation runs:

- **Unsupported capability claims**  
  The agent suggested actions (e.g. reminders) that were not available via its tools.

- **Off-scope persistence**  
  When prompted with unrelated topics (weather, Wi-Fi), the agent continued the conversation instead of refusing and stopping.

- **Non-converging dialogues**  
  Vague scenarios caused repeated back-and-forth until the max-turn limit was hit.

- **Final-turn handling bug (fixed)**  
  Earlier versions terminated before processing the final user message, dropping the last agent response.

- **Unprompted tool escalation**  
  Read-only requests occasionally led to the agent proposing create/update actions without explicit user intent.

These issues were identified by inspecting transcripts and Phoenix traces across repeated runs.

---

## High-level structure

- **Agent**: Google Calendar MCP agent (OpenAI Agents SDK)  
- **Controller**: multi-turn loop with stop conditions  
- **Synthetic user**: scenario-aware LLM simulator  
- **Tracing**: Phoenix / OpenTelemetry  
- **Storage**: SQLite session memory and filesystem logs  

The agent implementation is intentionally secondary to the evaluation logic.

---

## Relationship to the Google Calendar MCP Agent

This framework uses the **Google Calendar MCP Agent** as its test subject.

The agent repository contains:
- MCP server integration and Google Calendar tooling  
- full agent instructions and tool definitions  
- setup and usage documentation  

This repository assumes that background and focuses only on evaluation behaviour.

[📁 GitHub repo](https://github.com/david-revell/google-calendar-mcp)

---

## How to run

1. Clone this repository:
   ```
   git clone <this-repo-url>
   ```

2. Set the required environment variables:
   - OpenAI API key  
   - Phoenix configuration (if tracing is enabled)

3. Define evaluation scenarios in `scenarios.csv`.

4. Run the evaluation loop for a specific scenario.  
   For example, to run the scenario called `off_scope_wifi`:
   ```
   $env:SCENARIO_NAME="off_scope_wifi"; python mcp_calendar_agent.py
   ```

5. Inspect the outputs:
   - Conversation logs in `conversation_logs/`
   - Traces in the Phoenix UI

No UI is provided; evaluation is performed via logs and traces.

---

## Status

Exploratory evaluation framework used for iterative testing and debugging of agent behaviour.
