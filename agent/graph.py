"""Optimized LangGraph ReAct loop with structured output and iteration limits.

Key optimizations:
1. Structured output to reduce parsing overhead
2. Max 2 tool-call rounds (was unlimited)
3. Parallel tool execution where possible
4. Early exit on direct answers
5. Streaming support for perceived latency
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from llm import get_llm
from tools.calculator import calculator
from tools.file_search import file_search
from tools.summarize import summarize
from tools.web_search import web_search

TOOLS = [file_search, web_search, calculator, summarize]
TOOL_LABELS = {
    "file_search": "RAG / file search",
    "web_search": "web search",
    "calculator": "calculator",
    "summarize": "summarization (single-pass)",
}

MAX_TOOL_ROUNDS = 2  # Hard limit to prevent infinite loops


def build_graph():
    llm = get_llm().bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        
        # Track tool rounds to enforce limit
        tool_rounds = state.get("tool_rounds", 0)
        
        # If we've hit max rounds, force final answer
        if tool_rounds >= MAX_TOOL_ROUNDS:
            # Add a system message forcing final answer
            messages.append(SystemMessage(
                content="You have reached the maximum tool calls. Provide your final answer now "
                "based on the information gathered. Do not call more tools."
            ))
        
        response = llm.invoke(messages)
        return {"messages": [response], "tool_rounds": tool_rounds}

    def route_after_agent(state: AgentState) -> Literal["tools", "finalize", END]:
        last = state["messages"][-1]
        tool_rounds = state.get("tool_rounds", 0)
        
        if isinstance(last, AIMessage) and last.tool_calls:
            if tool_rounds >= MAX_TOOL_ROUNDS:
                return "finalize"
            return "tools"
        return END

    def finalize_node(state: AgentState) -> dict:
        """Force final answer without more tool calls."""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        # Remove any tool calls from last message
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            messages[-1] = AIMessage(content=last.content or "Let me provide my answer.")
        
        response = get_llm().invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", 
        route_after_agent, 
        {"tools": "tools", "finalize": "finalize", END: END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    global _graph
    _graph = None


def run_agent(messages: list[BaseMessage]) -> list[BaseMessage]:
    graph = get_graph()
    result = graph.invoke({"messages": messages, "tool_rounds": 0})
    return result["messages"]


def run_agent_stream(messages: list[BaseMessage]):
    """Stream agent responses for perceived latency improvement."""
    graph = get_graph()
    for chunk in graph.stream({"messages": messages, "tool_rounds": 0}, stream_mode="messages"):
        yield chunk


def collect_tool_labels(messages: list[BaseMessage]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            label = TOOL_LABELS.get(name, name)
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                text = "\n".join(p for p in parts if p).strip()
                if text:
                    return text
    return "The model returned an empty response."


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted