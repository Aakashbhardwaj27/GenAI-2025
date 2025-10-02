# AI Agents and Tools

AI agents are **programs that perform tasks autonomously** using LLMs, tools, and logic.

---

## What is an Agent?

- Receives a query or task  
- Decides which **tools or knowledge sources** to use  
- Executes actions to produce a result  

---

## Minimal vs Multi-Agent (MCP)

| Type | Description | Use Case |
|------|-------------|---------|
| Minimal agent | Single agent handling one task | Chatbot, simple automation |
| Multi-agent (MCP) | Multiple agents coordinating | Large systems, enterprise workflows |

---

## Tools Integration

- LLMs can call external tools:  
  - Google search  
  - Database queries  
  - Internal APIs  

---

## Simple Example Workflow

```text
User Query: "Summarize latest sales data"
1. Agent queries internal database
2. Agent formats results
3. LLM generates summary
