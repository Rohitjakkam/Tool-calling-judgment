# app.py
"""
Legal Judgment Search Application

This application provides a legal research agent with 24 specialized tools
for searching Indian legal judgments in Elasticsearch.

Features:
- Single-stage mode: GPT-4o agent for tool calling and responses
- Pipeline mode: GPT-4o retrieval + Gemini 2.5 Flash for detailed responses

Usage:
    python app.py              # Interactive mode (pipeline)
    python app.py --single     # Single agent mode
    python app.py --demo       # Run demo
    python app.py --tools      # List all tools

Or import and use programmatically:
    from app import search, agent_query, pipeline_query
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
    search_bail_cases,
    search_quashing_cases,
)
from tools.agent import create_legal_agent, run_query, LegalResearchAgent
from tools.response_agent import LegalResearchPipeline


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
        "bail": search_bail_cases,
        "quashing": search_quashing_cases,
    }

    tool = tools_map.get(tool_name, hybrid_search)
    return tool.invoke({"query": query, **kwargs})


def agent_query(query: str, model: str = "gpt-4o") -> dict:
    """
    Query the legal research agent (auto-selects best tool).
    Single-stage: uses GPT-4o for both retrieval and response.

    Args:
        query: Natural language query
        model: OpenAI model to use

    Returns:
        Dictionary with answer and tools used
    """
    agent = create_legal_agent(model=model, verbose=False)
    return run_query(agent, query)


def pipeline_query(
    query: str,
    retrieval_model: str = "gpt-4o",
    response_model: str = "gemini-2.5-flash",
    verbose: bool = True
) -> dict:
    """
    Query the two-stage legal research pipeline.
    Stage 1: GPT-4o retrieves data from Elasticsearch
    Stage 2: Gemini 2.5 Flash generates detailed response

    Args:
        query: Natural language query
        retrieval_model: Model for retrieval (tool calling)
        response_model: Model for response generation
        verbose: Whether to print intermediate steps

    Returns:
        Dictionary with raw_results, detailed_answer, and tools_used
    """
    pipeline = LegalResearchPipeline(
        retrieval_model=retrieval_model,
        response_model=response_model,
        verbose=verbose
    )
    return pipeline.query(query)


def print_tools_info():
    """Print information about all available tools."""
    print("\n" + "=" * 60)
    print("AVAILABLE LEGAL SEARCH TOOLS (24 Total)")
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


def interactive_mode(use_pipeline: bool = True):
    """
    Run the application in interactive mode.

    Args:
        use_pipeline: If True, use two-stage pipeline (GPT + Gemini)
                      If False, use single-stage agent (GPT only)
    """
    print("\n" + "=" * 60)
    print("LEGAL JUDGMENT SEARCH - Interactive Mode")
    print("=" * 60)

    if use_pipeline:
        print("\nMode: Two-Stage Pipeline")
        print("  Stage 1: GPT-4o (Retrieval & Tool Selection)")
        print("  Stage 2: Gemini 2.5 Flash (Detailed Response)")
    else:
        print("\nMode: Single Agent (GPT-4o)")

    print("\nCommands:")
    print("  /tools    - List all available tools")
    print("  /direct   - Use direct tool search (no agent)")
    print("  /agent    - Use single agent mode (GPT-4o only)")
    print("  /pipeline - Use pipeline mode (GPT + Gemini)")
    print("  /clear    - Clear conversation history")
    print("  /quit     - Exit the application")

    mode = "pipeline" if use_pipeline else "agent"
    agent = None
    pipeline = None

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
                    print("Switched to single agent mode (GPT-4o only).")
                    continue
                elif cmd == "/pipeline":
                    mode = "pipeline"
                    pipeline = None  # Reset pipeline
                    print("Switched to pipeline mode (GPT + Gemini).")
                    continue
                elif cmd == "/clear":
                    if agent:
                        agent.clear_history()
                    if pipeline:
                        pipeline.clear_history()
                    print("Conversation history cleared.")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Process query
            if mode == "pipeline":
                if pipeline is None:
                    print("\nInitializing pipeline...")
                    print("  - Loading GPT-4o retrieval agent...")
                    print("  - Loading Gemini 2.5 Flash response agent...")
                    pipeline = LegalResearchPipeline(verbose=True)
                    print("Pipeline ready!\n")

                result = pipeline.query(user_input)
                print(f"\n{'=' * 60}")
                print("DETAILED RESPONSE")
                print("=" * 60)
                print(f"\n{result['detailed_answer']}")

            elif mode == "agent":
                if agent is None:
                    print("Initializing agent...")
                    agent = LegalResearchAgent(model="gpt-4o", verbose=True)

                result = agent.query(user_input)
                print(f"\nTools used: {result['tools_used']}")
                print(f"\nAnswer:\n{result['answer']}")

            else:  # direct mode
                print("\nAvailable tools: party_names, act_section, citation, court,")
                print("                 topic, judge, date_range, hybrid, landmark,")
                print("                 stats, bail, quashing")
                tool_name = input("Select tool (default: hybrid): ").strip() or "hybrid"
                result = search(user_input, tool_name)
                print(f"\nResults:\n{result}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


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
    print("Searching for bail cases using specialized bail tool...")

    result = search_bail_cases.invoke({
        "bail_type": "anticipatory",
        "court": "supreme court",
        "size": 3
    })
    print(result)

    # Demo 3: Agent-based search
    print("\n### Demo 3: Agent-based Search ###")
    print("The agent automatically selects the best tool for your query.")
    print("Example: agent_query('Find landmark cases on right to privacy')")

    # Demo 4: Pipeline search
    print("\n### Demo 4: Pipeline Search ###")
    print("Two-stage pipeline: GPT-4o retrieval + Gemini response.")
    print("Example: pipeline_query('Anticipatory bail under Section 420 IPC')")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo()
        elif sys.argv[1] == "--tools":
            print_tools_info()
        elif sys.argv[1] == "--single":
            interactive_mode(use_pipeline=False)
        elif sys.argv[1] == "--help":
            print("""
Legal Judgment Search Application

Usage:
    python app.py              # Interactive mode with pipeline (GPT + Gemini)
    python app.py --single     # Interactive mode with single agent (GPT only)
    python app.py --demo       # Run demo
    python app.py --tools      # List all tools
    python app.py --help       # Show this help

Interactive Commands:
    /tools    - List all available tools
    /direct   - Use direct tool search (no agent)
    /agent    - Use single agent mode (GPT-4o only)
    /pipeline - Use pipeline mode (GPT + Gemini)
    /clear    - Clear conversation history
    /quit     - Exit the application

Programmatic Usage:
    from app import search, agent_query, pipeline_query

    # Direct search
    result = search("bail", tool_name="bail")

    # Single agent search (GPT-4o)
    result = agent_query("Find Section 138 NI Act cases from Supreme Court")

    # Pipeline search (GPT-4o + Gemini)
    result = pipeline_query("Anticipatory bail under Section 420 IPC after 2020")

Environment Variables Required:
    OPENAI_API_KEY     - For GPT-4o retrieval agent
    GOOGLE_API_KEY     - For Gemini response agent (pipeline mode only)
""")
        else:
            # Treat as a query - use pipeline by default
            query = " ".join(sys.argv[1:])
            print("Processing query with pipeline...")
            result = pipeline_query(query, verbose=False)
            print(f"\n{result['detailed_answer']}")
    else:
        interactive_mode(use_pipeline=True)
