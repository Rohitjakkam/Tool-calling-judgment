# tools/agent.py
"""
Legal Research Agent using all search tools.
Uses the modern LangChain create_agent pattern.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List, Dict, Any

from . import get_all_tools


# System prompt for the legal research agent
SYSTEM_PROMPT = """You are an expert legal research assistant with access to a comprehensive Indian legal judgment database.

You have access to 20 specialized search tools. Choose the most appropriate tool based on the query.

## Tool Selection Guide:

### For Party/Case Name Queries:
- `search_by_party_names` → When both petitioner AND respondent are mentioned (e.g., "Ram vs Shyam")
- `search_by_single_party` → When only one party name is mentioned (e.g., "cases involving Tata")
- `fuzzy_name_search` → When names might be misspelled or partially known

### For Legal Reference Queries:
- `search_by_act_section` → For specific sections (e.g., "Section 138 NI Act")
- `search_by_citation` → For case citations (e.g., "AIR 2020 SC 123")
- `search_by_multiple_sections` → When multiple sections are mentioned. IMPORTANT: Use the `topic` parameter when user asks for a specific subject like "anticipatory bail", "quashing", "discharge" along with sections (e.g., "anticipatory bail under Section 420, 467, 468 IPC")
- `search_by_legal_principle` → For doctrines/principles (e.g., "res judicata")

### For Court/Judge Queries:
- `search_by_court` → For court-specific searches (e.g., "Supreme Court cases")
- `search_by_judge` → For judge-specific searches (e.g., "judgments by Justice Chandrachud")
- `search_by_date_range` → For time-bound searches (e.g., "cases from 2018-2022")
- `search_by_bench_size` → For bench composition (e.g., "constitution bench judgments")

### For Topic/Content Queries:
- `search_by_legal_topic` → For subject areas (e.g., "bail cases", "murder judgments")
- `search_by_keywords` → For general keyword searches
- `search_by_case_type` → For case types (e.g., "writ petitions", "SLPs")
- `advanced_boolean_search` → For complex AND/OR/NOT queries

### For Advanced Queries:
- `search_similar_cases` → To find related/similar judgments
- `hybrid_search` → For complex natural language queries (fallback)
- `aggregation_search` → For statistics and counts
- `search_landmark_cases` → For landmark/important judgments
- `search_by_case_status` → To check if cases are overruled/followed

## Response Guidelines:
1. Always explain which tool you're using and why
2. If no results found, try alternative tools or suggest query modifications
3. Summarize key findings from the judgments
4. Mention case names, citations, and courts in your response
5. If the query is ambiguous, ask for clarification

Remember: Choose the MOST SPECIFIC tool that matches the query. Use `hybrid_search` only as a last resort.
"""


def create_legal_agent(
    model: str = "gpt-4o",
    temperature: float = 0,
    verbose: bool = True
):
    """
    Create a legal research agent with all available tools.

    Args:
        model: OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)
        temperature: LLM temperature (0 for deterministic)
        verbose: Whether to print agent steps

    Returns:
        Agent ready to process queries
    """
    # Get all tools
    tools = get_all_tools()

    # Create LLM
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Create agent using the new create_agent pattern
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_query(
    agent,
    query: str,
    chat_history: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run a query through the agent.

    Args:
        agent: Agent instance
        query: User query
        chat_history: Optional chat history for context

    Returns:
        Dictionary with output and tools used
    """
    # Build messages
    messages = []

    # Add chat history if provided
    if chat_history:
        messages.extend(chat_history)

    # Add current query
    messages.append({"role": "user", "content": query})

    # Invoke agent
    result = agent.invoke({"messages": messages})

    # Extract the answer from the result
    output_messages = result.get("messages", [])
    answer = ""
    tools_used = []

    for msg in output_messages:
        if hasattr(msg, 'content') and msg.content:
            # Get the last AI message as the answer
            if hasattr(msg, 'type') and msg.type == 'ai':
                answer = msg.content
            elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                answer = msg.get('content', '')

        # Track tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tools_used.append(tool_call.get('name', ''))
                elif hasattr(tool_call, 'name'):
                    tools_used.append(tool_call.name)

    # If answer is still empty, try to get from last message
    if not answer and output_messages:
        last_msg = output_messages[-1]
        if hasattr(last_msg, 'content'):
            answer = last_msg.content
        elif isinstance(last_msg, dict):
            answer = last_msg.get('content', str(last_msg))

    return {
        "answer": answer,
        "tools_used": tools_used,
        "messages": output_messages
    }


class LegalResearchAgent:
    """
    Wrapper class for the legal research agent with conversation memory.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        verbose: bool = True
    ):
        """
        Initialize the legal research agent.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature
            verbose: Whether to print agent steps
        """
        self.agent = create_legal_agent(model, temperature, verbose)
        self.chat_history: List = []

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query and maintain conversation history.

        Args:
            user_input: User's question or request

        Returns:
            Dictionary with answer, tools used, and messages
        """
        result = run_query(self.agent, user_input, self.chat_history)

        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": result["answer"]})

        return result

    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []

    def get_history(self) -> List:
        """Get the current conversation history."""
        return self.chat_history


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("LEGAL RESEARCH AGENT - Interactive Mode")
    print("=" * 60)
    print("\nInitializing agent...")

    try:
        agent = LegalResearchAgent(model="gpt-4o", verbose=True)
        print("Agent ready! Type 'quit' to exit.\n")

        while True:
            query = input("\nYour query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            try:
                result = agent.query(query)
                print(f"\n{'='*40}")
                print(f"Tools used: {result['tools_used']}")
                print(f"{'='*40}")
                print(f"\nAnswer:\n{result['answer']}")
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Failed to initialize agent: {e}")
