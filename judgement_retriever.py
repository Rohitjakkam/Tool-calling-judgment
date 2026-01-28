import os
from langchain_core.documents import Document
from typing import List
from collections import Counter
import time
from pprint import pprint
from elasticsearch import Elasticsearch
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
import time
from pprint import pprint
es_url = "http://139.84.219.174:9200"
def get_es_client(max_retries: int = 1, sleep_time: int = 5) -> Elasticsearch:
    i = 0
    while i < max_retries:
        try:
            es = Elasticsearch(es_url,request_timeout=60)
            pprint("Connected to Elasticsearch!")
            return es
        except Exception:
            pprint("Could not connect to Elasticsearch, retrying...")
            time.sleep(sleep_time)
            i += 1
    raise ConnectionError("Failed to connect to Elasticsearch after multiple attempts.")
class CaseMetadata(BaseModel):
    court_name: Optional[str] = Field(
        default=str,
        description="Name of the court (supreme court, [location] high court, tribunal, etc.)"
    )
    petitioner_names: Optional[List[str]] = Field(
        default_factory=list,
        description="List of petitioner names extracted from the query"
    )
    respondent_names: Optional[List[str]] = Field(
        default_factory=list,
        description="List of respondent names extracted from the query"
    )
    year: Optional[int] = Field(
        None,
        description="The 4-digit year mentioned in the query, if any"
    )
    topics: Optional[List[str]] = Field(
        default_factory=list,
        description="List of general legal topics or issues (e.g., 'murder', 'bail', 'negligence')"
    )
    acts_or_sections: Optional[List[str]] = Field(
        default_factory=list,
        description="List of legal provisions or sections mentioned (e.g., 'section 138 ni act', 'section 482 crpc')"
    )
    lexical_query: Optional[str] = Field(
        None,
        description="Normalized lexical search string constructed for Elasticsearch BM25 retrieval"
    )
    size: Optional[int] = Field(
        3,
        description=(
            "Number of results to retrieve from Elasticsearch. "
            "Defaults to 3. Set dynamically based on query specificity: "
            "1 for exact party match, 5 for section-based queries, 10 for topic-based queries."
        )
    )
 

prompt_template = """
You are an expert in **legal text parsing and Elasticsearch query formulation**.
 
The user may provide a natural language query that could include:
- A **case title** (with parties and/or year), or
- A **legal topic** (e.g., â€œcase laws on murderâ€, â€œjudgments under Section 420 IPCâ€), or
- A **court-specific request** (e.g., â€œSupreme Court judgmentsâ€, â€œBombay High Court bail casesâ€).
 
Your task is to extract **structured metadata** and generate a **normalized lexical search string**
that can be used for Elasticsearch BM25 retrieval.
 
---
 
### ðŸŽ¯ Expected JSON Output
Return output **strictly in valid JSON** with the following keys:
 
- court_name : Name of the court (string)
- petitioner_names: list of petitioners (strings)
- respondent_names: list of respondents (strings)
- year: 4-digit year (integer or null)
- topics: list of general legal topics (e.g., â€œmurderâ€, â€œnegligenceâ€)
- acts_or_sections: list of legal provisions (e.g., â€œsection 2 ipcâ€, â€œarticle 21 constitutionâ€)
- lexical_query: normalized string for lexical (BM25) search
- size: integer (number of documents to retrieve from Elasticsearch)
 
---
 
### âš–ï¸ Rules for Extraction
 
1. **Court Name Detection**
   - Identify mentions of courts such as:
     - "supreme court"
     - "[location] high court" (e.g., "Bombay High Court", "Delhi High Court")
     - "district court", "family court", "consumer court", "tribunal", "NCLT", "NCLAT", "ITAT", "CAT", "DRT", etc.
   - Recognize abbreviations:
     - â€œSCâ€ â†’ â€œsupreme courtâ€
     - â€œHCâ€ or â€œDelhi HCâ€ â†’ â€œdelhi high courtâ€
   - Normalize to lowercase and remove trailing words like â€œof Indiaâ€ or â€œatâ€.
     - Example: â€œSupreme Court of Indiaâ€ â†’ â€œsupreme courtâ€
     - Example: â€œHigh Court of Delhiâ€ â†’ â€œdelhi high courtâ€
   - If no court is mentioned, output as an empty string (`""`).
 
2. The party appearing **before** â€œVersusâ€, â€œVsâ€, or â€œv.â€ â†’ petitioner(s).  
   The party **after** â†’ respondent(s).
 
3. Preserve multi-word names (e.g., â€œVed Prakash & Ors.â€) â€” do **not** split them.
 
4. Normalize â€œv.â€, â€œvsâ€, â€œVsâ€ â†’ â€œversusâ€.
 
5. If no parties are detected:
   - Identify main **legal topic(s)** or **statutory references**.
   - Use those to populate `topics` and `acts_or_sections`.
 
6. If a legal provision is mentioned (e.g., â€œSection 2 of IPCâ€, â€œOrder 7 Rule 11 CPCâ€):
   - Extract under `acts_or_sections`.
   - Normalize to lowercase, remove â€œofâ€, â€œunderâ€, punctuation, and join with spaces.
     - Example: â€œSection 482 of CrPCâ€ â†’ â€œsection 482 crpcâ€.
 
7. Extract general legal subjects under `topics` (e.g., â€œmurderâ€, â€œbailâ€, â€œdefamationâ€).
 
8. **Construct the `lexical_query`**:
   - If both petitioner and respondent exist â†’  
     `"{{petitioner_names}} versus {{respondent_names}} {{year}}"`.
   - If both `acts_or_sections` and `topics` exist â†’  
     `"{{acts_or_sections}} {{topics}}"`.
   - If only one of them exists, use it directly.
   - Include `court_name` at the start of `lexical_query` if present, **unless** it already appears in party names.
   - Lowercase, remove punctuation, and collapse multiple spaces.
 
9. **Remove filler or intent words** like â€œgive meâ€, â€œfindâ€, â€œshowâ€, â€œtellâ€, â€œlistâ€, â€œlatestâ€, â€œrecentâ€, â€œimportantâ€, â€œfamousâ€, â€œlandmarkâ€, etc.
 
10. **Normalize `lexical_query`:**
    - Convert to lowercase, strip punctuation, and remove duplicate spaces.
 
11. **Explicit `size` detection:**
    - If the user mentions numbers (â€œtop 5â€, â€œshow 10â€, â€œfirst 20â€, â€œonly 1â€), extract that integer as `size`.
    - Words like â€œoneâ€, â€œsingleâ€, or â€œbestâ€ â†’ `size = 1`.
    - Clamp to [1, 50]: if user asks for >50 â†’ set to 50.
 
12. **Heuristic `size` determination** (only if no explicit size found):
    - Both petitioner & respondent present â†’ `size = 1`
    - If `acts_or_sections` exist â†’ `size = 5`
    - If only `topics` exist â†’ `size = 10`
    - Otherwise â†’ `size = 3` (default)
 
13. **Conflict rule:**  
    If both explicit and heuristic rules match, the **explicit user request overrides** the heuristic.
 
14. If a field cannot be found, output it as:
    - `""` for strings  
    - `[]` for lists  
    - `null` for year
 
15. **Output Format Enforcement**
    - Output only valid JSON (no markdown, no ```json``` fences).
    - Maintain field order as in Expected JSON Output.
 
---
 
### ðŸ§  Examples
 
**Example 1 â€” Court + Statutory Reference**
> "Supreme Court judgments under Section 138 of NI Act"
 
**Expected JSON output:**
{{
  "court_name": "supreme court",
  "petitioner_names": [],
  "respondent_names": [],
  "year": null,
  "topics": [],
  "acts_or_sections": ["section 138 ni act"],
  "lexical_query": "supreme court section 138 ni act",
  "size": 5
}}
 
---
 
**Example 2 â€” High Court + Topic**
> "Bombay High Court cases on negligence"
 
**Expected JSON output:**
{{
  "court_name": "bombay high court",
  "petitioner_names": [],
  "respondent_names": [],
  "year": null,
  "topics": ["negligence"],
  "acts_or_sections": [],
  "lexical_query": "bombay high court negligence",
  "size": 10
}}
 
---
 
**Example 3 â€” Case Title with Year**
> "State of Maharashtra Vs John Doe 2020"
 
**Expected JSON output:**
{{
  "court_name": "",
  "petitioner_names": ["State of Maharashtra"],
  "respondent_names": ["John Doe"],
  "year": 2020,
  "topics": [],
  "acts_or_sections": [],
  "lexical_query": "state of maharashtra versus john doe 2020",
  "size": 1
}}
 
---
 
**User query:**  
{query}
 
**Output only valid JSON.**
"""
 
 
 
 
def query_metadata_llm(query: str):
    llm = ChatOpenAI(temperature=0.3, model="gpt-4o").with_structured_output(CaseMetadata)
    QUERY_PROMPT = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = QUERY_PROMPT | llm
    res = llm_chain.invoke({'query': query})
    return res
 
def build_query(metadata, query_text=None, size=1):
    """
    Build an Elasticsearch BM25 query for legal judgments using multiple tiers:
   
    Tiers:
        1. Party-based exact match (if petitioner/respondent names exist)
        2. Fuzzy match (for slightly misspelled names)
        3. Keyword/topic-based search (from 'topics')
        4. Statutory reference search (from 'acts_or_sections')
        5. Lexical fallback (from LLM-generated lexical_query)
        6. Full-text fallback (page_content)
   
    Filters:
        - year (if provided)
    """
    should_clauses = []
    filters = []
 
    # Extract metadata fields
    petitioner_names = metadata.petitioner_names or []
    respondent_names = metadata.respondent_names or []
    topics = metadata.topics or []
    acts_or_sections = metadata.acts_or_sections or []
    lexical_query = metadata.lexical_query or query_text
 
    # ---------------------------
    # TIER 1: Exact Party Match
    # ---------------------------
    if petitioner_names and respondent_names:
        for petitioner in petitioner_names:
            for respondent in respondent_names:
                should_clauses.append({
                    "bool": {
                        "must": [
                            {"match": {"petitioner_names": {"query": petitioner, "boost": 20, "minimum_should_match": "95%"}}},
                            {"match": {"respondent_names": {"query": respondent, "boost": 20, "minimum_should_match": "95%"}}}
                        ]
                    }
                })
 
    # ---------------------------
    # TIER 2: Fuzzy Party Match
    # ---------------------------
    # if petitioner_names and respondent_names:
    #     for petitioner in petitioner_names:
    #         for respondent in respondent_names:
    #             should_clauses.append({
    #                 "bool": {
    #                     "must": [
    #                         {"match": {"petitioner_names": {"query": petitioner, "fuzziness": "AUTO", "boost": 15}}},
    #                         {"match": {"respondent_names": {"query": respondent, "fuzziness": "AUTO", "boost": 15}}}
    #                     ]
    #                 }
    #             })
 
    # ---------------------------
    # TIER 3: Topics (Legal Issues)
    # ---------------------------
    for topic in topics:
        should_clauses.append({
            "match": {"keywords": {"query": topic, "boost": 15, "minimum_should_match": "95%"}}
        })
 
    # ---------------------------
    # TIER 4: Acts / Sections Invoked
    # ---------------------------
    for section in acts_or_sections:
        should_clauses.append({
            "match": {"acts_or_sections_invoked": {"query": section, "boost": 15, "minimum_should_match": "95%"}}
        })
 
    # ---------------------------
    # TIER 5: Lexical Query (LLM normalized)
    # ---------------------------
    if lexical_query:
        should_clauses.append({
            "match": {"page_content": {"query": lexical_query, "boost": 8, "minimum_should_match": "95%"}}
        })
 
    # ---------------------------
    # TIER 6: Fallback - full text
    # ---------------------------
    if query_text and not lexical_query:
        should_clauses.append({
            "match": {"page_content": {"query": query_text, "boost": 3,}}
        })
 
    # ---------------------------
    # Filters
    # ---------------------------
    if metadata.year:
        filters.append({"term": {"year": metadata.year}})
    if metadata.court_name:
        filters.append({"term": {"court_name": metadata.court_name.lower()}})
 
    # ---------------------------
    # Final Query Body
    # ---------------------------
    query_body = {
        "size": metadata.size if metadata.size else size,
        "query": {
            "bool": {
                "should": should_clauses,
                "filter": filters if filters else [],
                "minimum_should_match": 1
            }
        }
    }
 
    return query_body
 
class JudgementRetriever:
    def __init__(self):
        pass
    def retrieve_documents(self,query: str,top_k=17) -> List[Document]:
        try: 
            query_metadata = query_metadata_llm(query)
            print(query_metadata)
    
            es_query = build_query(metadata= query_metadata,query_text= query,)
            print(es_query)
            es = get_es_client()
            # es_with_option = es.with_options(request_timeout=60)
            # es_with_option = Elasticsearch(es_url, request_timeout=60)

            source_response = es.options(request_timeout=50).search(index="judgements", body=es_query)
            langchain_docs = []
            for hit in source_response["hits"]["hits"]:
                content = hit["_source"]["page_content"]
                source = {"source": hit["_source"]["source"]}
                court = hit["_source"]["court_name"]
                respondent_names = hit["_source"]["respondent_names"][0]
                petitioner_names = hit["_source"]["petitioner_names"][0]
                title = f"{petitioner_names} vs {respondent_names}"
                metadata = {"source": source, "source_name": title,"court":court}
                doc = Document(page_content=content, metadata=metadata)
                langchain_docs.append(doc)
            print(langchain_docs)
            return langchain_docs
        except Exception as e:
            print(f"âš ï¸ Hybrid search failed: {e}")
            return []