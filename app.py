# app.py
"""
Legal Judgment Search Application

This application provides a legal research agent with 20 specialized tools
for searching Indian legal judgments in Elasticsearch.

Usage:
    python app.py

Or import and use programmatically:
    from app import search, agent_query
"""

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import tools and agent
from tools import (
    get_tools_by_category,
    # Individual tools for direct use
    search_by_party_names,
    search_by_act_section,
    search_by_citation,
    search_by_court,
    search_by_legal_topic,
    search_by_judge,
    search_by_date_range,
    hybrid_search,
    search_landmark_cases,
    aggregation_search,
)
from tools.agent import create_legal_agent, run_query, LegalResearchAgent


def search(query: str, tool_name: str = "hybrid_search", **kwargs) -> str:
    """
    Search for legal judgments using a specific tool.

    Args:
        query: Search query
        tool_name: Name of the tool to use
        **kwargs: Additional arguments for the tool

    Returns:
        Search results as formatted string
    """
    tools_map = {
        "party_names": search_by_party_names,
        "act_section": search_by_act_section,
        "citation": search_by_citation,
        "court": search_by_court,
        "topic": search_by_legal_topic,
        "judge": search_by_judge,
        "date_range": search_by_date_range,
        "hybrid": hybrid_search,
        "landmark": search_landmark_cases,
        "stats": aggregation_search,
    }

    tool = tools_map.get(tool_name, hybrid_search)
    return tool.invoke({"query": query, **kwargs})


def agent_query(query: str, model: str = "gpt-4o") -> dict:
    """
    Query the legal research agent (auto-selects best tool).

    Args:
        query: Natural language query
        model: OpenAI model to use

    Returns:
        Dictionary with answer and tools used
    """
    agent = create_legal_agent(model=model, verbose=False)
    return run_query(agent, query)


def print_tools_info():
    """Print information about all available tools."""
    print("\n" + "=" * 60)
    print("AVAILABLE LEGAL SEARCH TOOLS (20 Total)")
    print("=" * 60)

    categories = get_tools_by_category()

    for category, tools in categories.items():
        print(f"\n## {category.replace('_', ' ').title()} ({len(tools)} tools)")
        print("-" * 40)
        for tool in tools:
            print(f"  - {tool.name}")
            # Get first line of description
            desc = tool.description.split('\n')[0][:60]
            print(f"    {desc}...")


def interactive_mode():
    """Run the application in interactive mode."""
    print("\n" + "=" * 60)
    print("LEGAL JUDGMENT SEARCH - Interactive Mode")
    print("=" * 60)

    print("\nCommands:")
    print("  /tools    - List all available tools")
    print("  /direct   - Use direct tool search (no agent)")
    print("  /agent    - Use agent mode (auto-selects tool)")
    print("  /clear    - Clear conversation history")
    print("  /quit     - Exit the application")

    mode = "agent"
    agent = None

    while True:
        try:
            user_input = input(f"\n[{mode}] Query: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break
                elif cmd == "/tools":
                    print_tools_info()
                    continue
                elif cmd == "/direct":
                    mode = "direct"
                    print("Switched to direct tool mode.")
                    continue
                elif cmd == "/agent":
                    mode = "agent"
                    agent = None  # Reset agent
                    print("Switched to agent mode.")
                    continue
                elif cmd == "/clear":
                    if agent:
                        agent.clear_history()
                    print("Conversation history cleared.")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Process query
            if mode == "agent":
                if agent is None:
                    print("Initializing agent...")
                    agent = LegalResearchAgent(model="gpt-4o", verbose=True)

                result = agent.query(user_input)
                print(f"\nTools used: {result['tools_used']}")
                print(f"\nAnswer:\n{result['answer']}")

            else:  # direct mode
                print("\nAvailable tools: party_names, act_section, citation, court,")
                print("                 topic, judge, date_range, hybrid, landmark, stats")
                tool_name = input("Select tool (default: hybrid): ").strip() or "hybrid"
                result = search(user_input, tool_name)
                print(f"\nResults:\n{result}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {e}")


# Example usage demonstrations
def demo():
    """Demonstrate the various ways to use the legal search tools."""
    print("\n" + "=" * 60)
    print("LEGAL SEARCH TOOLS - DEMO")
    print("=" * 60)

    # Demo 1: Direct tool usage
    print("\n### Demo 1: Direct Tool Usage ###")
    print("Searching for Section 138 NI Act cases...")

    result = search_by_act_section.invoke({
        "act_name": "NI Act",
        "section": "138",
        "court": "supreme court",
        "size": 3
    })
    print(result)

    # Demo 2: Using the search helper
    print("\n### Demo 2: Search Helper ###")
    print("Searching for bail cases using topic search...")

    result = search("bail", tool_name="topic")
    print(result)

    # Demo 3: Agent-based search (commented out to avoid API calls in demo)
    print("\n### Demo 3: Agent-based Search ###")
    print("The agent automatically selects the best tool for your query.")
    print("Example: agent_query('Find landmark cases on right to privacy')")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo()
        elif sys.argv[1] == "--tools":
            print_tools_info()
        elif sys.argv[1] == "--help":
            print("""
Legal Judgment Search Application

Usage:
    python app.py              # Interactive mode
    python app.py --demo       # Run demo
    python app.py --tools      # List all tools
    python app.py --help       # Show this help

Programmatic Usage:
    from app import search, agent_query

    # Direct search
    result = search("bail", tool_name="topic")

    # Agent search (auto-selects tool)
    result = agent_query("Find Section 138 NI Act cases from Supreme Court")
""")
        else:
            # Treat as a query
            query = " ".join(sys.argv[1:])
            result = agent_query(query)
            print(f"Tools used: {result['tools_used']}")
            print(f"\n{result['answer']}")
    else:
        interactive_mode()
