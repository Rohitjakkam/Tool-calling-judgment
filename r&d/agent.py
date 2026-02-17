"""
LangGraph agent for Supreme Court of India judgment search.
Uses create_agent with InMemorySaver for conversation memory.
"""

import os
import uuid
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools import get_all_tools

SYSTEM_PROMPT = """You are an expert legal research assistant specializing in Supreme Court of India judgments.
You have access to a database of 44,000+ Supreme Court judgments spanning from 1950 to 2026.

## Your Available Tools:

1. **search_by_semantic** - AI-powered semantic search. Use when user describes a legal concept, topic, or principle in natural language. Best for broad conceptual queries.

2. **search_by_keyword** - Full-text keyword search (BM25). Use when user mentions specific legal terms, section numbers, act names, or exact phrases. Best for precise term matching.

3. **search_by_case_number** - Case number lookup. Use when user provides a specific case number (e.g., "Criminal Appeal No. 883 of 2020").

4. **search_by_party_name** - Party name search. Use when user asks about cases involving a specific person, company, or government entity.

5. **search_by_date_range** - Date range filter. Use when user asks for judgments from a specific time period. Dates are in DD-MM-YYYY format.

6. **search_by_judge** - Judge/bench search. Use when user asks about cases decided by a specific judge. Note: judge information is more complete for cases from 2014 onwards.

7. **get_case_details** - Get full case details by database ID. Use AFTER finding a case through other tools to get complete information including all PDF links and full judgment text.

## Tool Selection Guidelines:

- For "find cases about [topic]" → use search_by_semantic first, then search_by_keyword if needed
- For "Section 302 IPC cases" → use search_by_keyword (exact terms)
- For "Kesavananda Bharati vs State of Kerala" → use search_by_party_name
- For "recent judgments on bail" → use search_by_semantic with year_from filter
- For "cases by Justice Chandrachud" → use search_by_judge
- For "Criminal Appeal No. 883/2020" → use search_by_case_number
- You may call multiple tools if needed to give a comprehensive answer

## Response Guidelines:

1. **Always cite specific cases** with their full party names and judgment dates
2. **Always provide PDF links** when available - format them clearly
3. **Summarize key holdings** from each relevant case
4. **Structure responses clearly** with headers and bullet points
5. If a query is ambiguous, ask for clarification
6. If no results found with one tool, try another approach before saying no results exist
7. When mentioning a case, include its DB ID so the user can request full details
8. For follow-up questions, use get_case_details to dive deeper into specific cases

## IMPORTANT:
- Never fabricate case names, citations, or holdings
- Only cite cases that appear in your search results
- If results are insufficient, say so honestly and suggest refining the query
- When providing PDF links, present them clearly so users can click to download
"""


class SCIJudgmentAgent:
    """Supreme Court judgment search agent with conversation memory."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.tools = get_all_tools()
        self.memory = InMemorySaver()
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
        )
        self.thread_id = str(uuid.uuid4())

    def query(self, user_input: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query. Uses thread_id for conversation continuity."""
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Extract answer and tools used
        messages = result.get("messages", [])
        answer = ""
        tools_used = []

        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                if hasattr(msg, "type") and msg.type == "ai":
                    answer = msg.content
                elif isinstance(msg, dict) and msg.get("role") == "assistant":
                    answer = msg.get("content", "")

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tools_used.append(name)

        # Fallback: get last message content
        if not answer and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content
            elif isinstance(last_msg, dict):
                answer = last_msg.get("content", str(last_msg))

        return {
            "answer": answer,
            "tools_used": tools_used,
            "messages": messages,
        }

    def stream(self, user_input: str, thread_id: Optional[str] = None):
        """Stream agent response for real-time UI updates."""
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            yield chunk

    def new_conversation(self) -> str:
        """Start a new conversation thread."""
        self.thread_id = str(uuid.uuid4())
        return self.thread_id


# Interactive CLI mode
if __name__ == "__main__":
    print("=" * 60)
    print("SCI JUDGMENT SEARCH AGENT - Interactive Mode")
    print("=" * 60)
    print("\nInitializing agent...")

    try:
        agent = SCIJudgmentAgent(model="gpt-4o")
        print("Agent ready! Type 'quit' to exit, 'new' for new conversation.\n")

        while True:
            query = input("\nYour query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "new":
                agent.new_conversation()
                print("New conversation started.")
                continue

            if not query:
                continue

            try:
                result = agent.query(query)
                print(f"\n{'=' * 40}")
                print(f"Tools used: {result['tools_used']}")
                print(f"{'=' * 40}")
                print(f"\n{result['answer']}")
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Failed to initialize agent: {e}")
