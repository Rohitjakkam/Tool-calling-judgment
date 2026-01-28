# tools/response_agent.py
"""
Response Agent using Gemini 2.5 Flash for generating detailed legal answers.
Takes raw search results and creates well-formatted, comprehensive responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional
import os


# System prompt for the response agent
RESPONSE_SYSTEM_PROMPT = """You are an expert Indian legal research analyst. Your task is to analyze legal search results and provide a comprehensive, well-structured response to the user's query.

## Your Role:
- Analyze the retrieved case data thoroughly
- Provide accurate legal information with proper citations
- Structure your response clearly for legal professionals
- Highlight key legal principles and holdings from the cases

## Response Format:
Always structure your response as follows:

### Summary
A brief 2-3 sentence overview answering the user's query directly.

### Relevant Case Laws
For each relevant case, provide:
- **Case Name**: [Petitioner vs Respondent]
- **Citation**: [Full citation if available]
- **Court**: [Court name]
- **Year**: [Year of judgment]
- **Key Holdings**: [Main legal principles established]
- **Relevance**: [Why this case is relevant to the query]
- **Document Link**: [S3 link if available]

### Legal Principles Established
Summarize the key legal principles that emerge from these cases.

### Practical Implications
Explain what these judgments mean for similar cases.

### Conclusion
A concise conclusion with recommendations if applicable.

## Important Guidelines:
1. ONLY use information from the provided search results - do not make up cases or citations
2. If the search results don't contain relevant cases, clearly state that
3. Always mention the court and year for each case
4. Use proper legal terminology
5. Be objective and accurate
6. If cases have conflicting views, present both perspectives
7. Highlight any landmark judgments separately
"""


class ResponseAgent:
    """
    Response agent that uses Gemini 2.5 Flash to generate detailed legal answers.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the response agent.

        Args:
            model: Gemini model to use
            temperature: LLM temperature (slightly higher for better prose)
            api_key: Google API key (optional, uses env var if not provided)
        """
        # Check for API key
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Google API key not found. Please either:\n"
                "1. Set GOOGLE_API_KEY in your .env file\n"
                "2. Pass api_key parameter to ResponseAgent\n"
                "3. Use single-agent mode: python app.py --single\n\n"
                "Get a free API key at: https://aistudio.google.com/apikey"
            )

        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=resolved_api_key
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RESPONSE_SYSTEM_PROMPT),
            ("human", """## User's Original Query:
{query}

## Search Results Retrieved:
{search_results}

## Tools Used for Retrieval:
{tools_used}

---

Please analyze the above search results and provide a comprehensive, well-structured response to the user's query. Focus on the cases that are most relevant to what the user asked for.""")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(
        self,
        query: str,
        search_results: str,
        tools_used: List[str]
    ) -> str:
        """
        Generate a detailed response based on search results.

        Args:
            query: User's original query
            search_results: Raw search results from retrieval agent
            tools_used: List of tools used during retrieval

        Returns:
            Well-formatted detailed response
        """
        tools_str = ", ".join(tools_used) if tools_used else "None"

        response = self.chain.invoke({
            "query": query,
            "search_results": search_results,
            "tools_used": tools_str
        })

        return response


class LegalResearchPipeline:
    """
    Two-stage pipeline combining retrieval agent and response agent.

    Stage 1: GPT-4o retrieval agent gathers data from Elasticsearch
    Stage 2: Gemini 2.5 Flash response agent creates detailed answers
    """

    def __init__(
        self,
        retrieval_model: str = "gpt-4o",
        response_model: str = "gemini-2.5-flash",
        retrieval_temperature: float = 0,
        response_temperature: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize the legal research pipeline.

        Args:
            retrieval_model: OpenAI model for retrieval (tool calling)
            response_model: Gemini model for response generation
            retrieval_temperature: Temperature for retrieval agent
            response_temperature: Temperature for response agent
            verbose: Whether to print intermediate steps
        """
        from .agent import LegalResearchAgent

        self.retrieval_agent = LegalResearchAgent(
            model=retrieval_model,
            temperature=retrieval_temperature,
            verbose=verbose
        )

        self.response_agent = ResponseAgent(
            model=response_model,
            temperature=response_temperature
        )

        self.verbose = verbose
        self.chat_history: List[Dict[str, str]] = []

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query through the two-stage pipeline.

        Args:
            user_input: User's question or request

        Returns:
            Dictionary with:
            - raw_results: Raw data from retrieval agent
            - detailed_answer: Polished response from Gemini
            - tools_used: List of tools used
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: Retrieving data from Elasticsearch...")
            print("=" * 60)

        # Stage 1: Retrieval
        retrieval_result = self.retrieval_agent.query(user_input)
        raw_results = retrieval_result.get("answer", "")
        tools_used = retrieval_result.get("tools_used", [])

        if self.verbose:
            print(f"\nTools used: {tools_used}")
            print(f"\nRaw results length: {len(raw_results)} characters")
            print("\n" + "=" * 60)
            print("STAGE 2: Generating detailed response with Gemini...")
            print("=" * 60)

        # Stage 2: Response Generation
        detailed_answer = self.response_agent.generate_response(
            query=user_input,
            search_results=raw_results,
            tools_used=tools_used
        )

        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": detailed_answer})

        return {
            "raw_results": raw_results,
            "detailed_answer": detailed_answer,
            "tools_used": tools_used
        }

    def clear_history(self):
        """Clear the conversation history."""
        self.retrieval_agent.clear_history()
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.chat_history


# Standalone function for quick use
def generate_legal_response(
    query: str,
    search_results: str,
    tools_used: List[str] = None,
    model: str = "gemini-2.5-flash"
) -> str:
    """
    Standalone function to generate a detailed legal response.

    Args:
        query: User's original query
        search_results: Raw search results
        tools_used: List of tools used (optional)
        model: Gemini model to use

    Returns:
        Detailed formatted response
    """
    agent = ResponseAgent(model=model)
    return agent.generate_response(
        query=query,
        search_results=search_results,
        tools_used=tools_used or []
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("LEGAL RESEARCH PIPELINE - Two-Stage Agent")
    print("Stage 1: GPT-4o (Retrieval)")
    print("Stage 2: Gemini 2.5 Flash (Response)")
    print("=" * 60)
    print("\nInitializing pipeline...")

    try:
        pipeline = LegalResearchPipeline(verbose=True)
        print("Pipeline ready! Type 'quit' to exit.\n")

        while True:
            query = input("\nYour query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            try:
                result = pipeline.query(query)
                print(f"\n{'=' * 60}")
                print("FINAL RESPONSE")
                print("=" * 60)
                print(f"\n{result['detailed_answer']}")
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
