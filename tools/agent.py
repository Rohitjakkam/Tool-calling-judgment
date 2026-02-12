# tools/agent.py
"""
Legal Research Agent using all search tools.
Uses the modern LangChain create_agent pattern.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List, Dict, Any

# Import tools directly to avoid circular import
from .party_search import search_by_party_names, fuzzy_name_search, search_by_single_party
from .legal_reference_search import search_by_act_section, search_by_citation, search_by_multiple_sections, search_by_legal_principle
from .court_search import search_by_court, search_by_judge, search_by_date_range, search_by_bench_size
from .content_search import search_by_legal_topic, search_by_keywords, advanced_boolean_search, search_by_case_type
from .advanced_search import search_similar_cases, hybrid_search, aggregation_search, search_landmark_cases, search_by_case_status
from .specialized_search import search_bail_cases, search_quashing_cases, search_writ_petitions, search_criminal_appeals
from .smart_search import smart_case_search


def get_all_tools_for_agent() -> List:
    """Get all tools for the agent (local function to avoid circular import)."""
    return [
        # Party Search Tools
        search_by_party_names, fuzzy_name_search, search_by_single_party,
        # Legal Reference Tools
        search_by_act_section, search_by_citation, search_by_multiple_sections, search_by_legal_principle,
        # Court Search Tools
        search_by_court, search_by_judge, search_by_date_range, search_by_bench_size,
        # Content Search Tools
        search_by_legal_topic, search_by_keywords, advanced_boolean_search, search_by_case_type,
        # Advanced Search Tools
        search_similar_cases, hybrid_search, aggregation_search, search_landmark_cases, search_by_case_status,
        # Specialized Search Tools
        search_bail_cases, search_quashing_cases, search_writ_petitions, search_criminal_appeals,
        # Smart Search (exhaustive multi-strategy)
        smart_case_search,
    ]


# System prompt for the legal research agent
SYSTEM_PROMPT = """You are an expert legal research assistant with access to a comprehensive Indian legal judgment database.

You have access to 25 specialized search tools. Choose the most appropriate tool based on the query.

## PRIORITY TOOL — Smart Case Search (USE FIRST for specific cases):

### `smart_case_search` — ALWAYS use this FIRST when searching for a SPECIFIC CASE
This tool automatically tries 7-12 different search strategies including:
- Both party names with/without year
- Swapped party names (petitioner↔respondent)
- Fuzzy name matching
- Name variations (strips "And Ors", "State of X" prefixes)
- Citation matching in citation field and page content
- Single-party search on each name individually
- Full-text keyword fallback

**USE `smart_case_search` when:**
- User gives a case name like "Ram vs Shyam"
- User gives a case name + citation like "Ram vs Shyam, (2005) 4 SCC 501"
- User gives just a citation like "AIR 2020 SC 123"
- User asks to "find" or "search for" a specific judgment
- Any query that names specific parties or a specific case

This single tool call replaces the need for 5-7 manual retries. It handles all permutations automatically.

## Other Tools (for non-specific-case queries):

### For Legal Reference Queries:
- `search_by_act_section` → For specific sections (e.g., "Section 138 NI Act")
- `search_by_multiple_sections` → When multiple sections are mentioned. Use the `topic` parameter when user asks for a specific subject along with sections.
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

### For Specialized Legal Topics (PREFERRED for these topics):
- `search_bail_cases` → **USE THIS** for bail queries (anticipatory bail, regular bail, interim bail). Supports sections, court, and year filters.
- `search_quashing_cases` → For Section 482 CrPC quashing petitions
- `search_writ_petitions` → For writ petitions (habeas corpus, mandamus, Article 226/32)
- `search_criminal_appeals` → For criminal appeals (conviction appeals, acquittal appeals)

### For Advanced Queries:
- `search_similar_cases` → To find related/similar judgments
- `hybrid_search` → For complex natural language queries (fallback)
- `aggregation_search` → For statistics and counts
- `search_landmark_cases` → For landmark/important judgments
- `search_by_case_status` → To check if cases are overruled/followed

### Individual Party/Citation Tools (only if smart_case_search fails):
- `search_by_party_names` → Both petitioner AND respondent
- `search_by_single_party` → Only one party name
- `fuzzy_name_search` → Misspelled or partial names
- `search_by_citation` → Case citations

## IMPORTANT Tool Selection Rules:
1. For SPECIFIC CASE lookups → ALWAYS use `smart_case_search` FIRST (it tries 7-12 strategies automatically)
2. For BAIL queries → Always use `search_bail_cases` first
3. For QUASHING queries → Always use `search_quashing_cases` first
4. For WRIT PETITION queries → Always use `search_writ_petitions` first
5. For multiple courts in same query → Make separate tool calls for each court
6. When year filter is specified → Always pass year_from/year_to parameters

## Response Guidelines:
1. Always explain which tool you're using and why
2. Summarize key findings from the judgments
3. Mention case names, citations, and courts in your response
4. If the query is ambiguous, ask for clarification

## CRITICAL: Never Give Up After One Try
- If `smart_case_search` returns no results, try `hybrid_search` as a last resort
- NEVER say "case not found" without trying at least `smart_case_search` + one more tool
- The database has corrupted year fields — never rely solely on year-based filtering

Remember: Use `smart_case_search` for specific cases. Use specialized tools (bail, quashing, writ, appeals) for topic-based queries.
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
    tools = get_all_tools_for_agent()

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
