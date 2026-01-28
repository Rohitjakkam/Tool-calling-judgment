# api/suggestions.py
"""
Query suggestions and quick actions for the legal research chatbot.
Provides pre-defined templates, popular queries, and smart suggestions.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QuickAction:
    """Represents a quick action button."""
    id: str
    label: str
    icon: str
    query_template: str
    description: str
    category: str


# Quick action buttons for common legal searches
QUICK_ACTIONS: List[QuickAction] = [
    # Bail related
    QuickAction(
        id="anticipatory_bail",
        label="Anticipatory Bail",
        icon="🛡️",
        query_template="Find anticipatory bail cases under Section {section} IPC from {court} after {year}",
        description="Search for anticipatory bail judgments",
        category="Bail"
    ),
    QuickAction(
        id="regular_bail",
        label="Regular Bail",
        icon="⚖️",
        query_template="Find regular bail cases under Section {section} IPC from {court}",
        description="Search for regular bail judgments",
        category="Bail"
    ),

    # Criminal
    QuickAction(
        id="murder_cases",
        label="Murder Cases",
        icon="🔍",
        query_template="Find murder cases under Section 302 IPC from {court} after {year}",
        description="Search for murder/homicide judgments",
        category="Criminal"
    ),
    QuickAction(
        id="cheating_cases",
        label="Cheating Cases",
        icon="💰",
        query_template="Find cheating cases under Section 420 IPC from {court}",
        description="Search for cheating/fraud judgments",
        category="Criminal"
    ),
    QuickAction(
        id="forgery_cases",
        label="Forgery Cases",
        icon="📝",
        query_template="Find forgery cases under Sections 467, 468, 471 IPC from {court}",
        description="Search for forgery related judgments",
        category="Criminal"
    ),

    # Procedural
    QuickAction(
        id="quashing_fir",
        label="Quashing of FIR",
        icon="❌",
        query_template="Find Section 482 CrPC quashing cases from {court} after {year}",
        description="Search for FIR quashing judgments",
        category="Procedural"
    ),
    QuickAction(
        id="discharge_cases",
        label="Discharge Petitions",
        icon="🚪",
        query_template="Find discharge petition cases under Section {section} from {court}",
        description="Search for discharge order judgments",
        category="Procedural"
    ),

    # Constitutional
    QuickAction(
        id="writ_petition",
        label="Writ Petitions",
        icon="📜",
        query_template="Find Article {article} writ petition cases from {court}",
        description="Search for writ petition judgments",
        category="Constitutional"
    ),
    QuickAction(
        id="fundamental_rights",
        label="Fundamental Rights",
        icon="🏛️",
        query_template="Find fundamental rights cases under Article {article} from Supreme Court",
        description="Search for fundamental rights judgments",
        category="Constitutional"
    ),

    # Civil
    QuickAction(
        id="contract_cases",
        label="Contract Disputes",
        icon="📋",
        query_template="Find contract dispute cases from {court} after {year}",
        description="Search for contract law judgments",
        category="Civil"
    ),
    QuickAction(
        id="property_cases",
        label="Property Matters",
        icon="🏠",
        query_template="Find property dispute cases from {court}",
        description="Search for property law judgments",
        category="Civil"
    ),

    # Landmark
    QuickAction(
        id="landmark_cases",
        label="Landmark Cases",
        icon="⭐",
        query_template="Find landmark judgments on {topic} from Supreme Court",
        description="Search for landmark/important judgments",
        category="Landmark"
    ),
]


# Popular/suggested queries
SUGGESTED_QUERIES = [
    {
        "query": "Anticipatory bail cases under Section 420, 467, 468 IPC from Supreme Court after 2020",
        "category": "Bail",
        "description": "Recent anticipatory bail cases for cheating and forgery"
    },
    {
        "query": "Section 138 NI Act cheque bounce cases from Supreme Court",
        "category": "Commercial",
        "description": "Negotiable Instruments Act cases"
    },
    {
        "query": "Article 21 right to life cases from Supreme Court",
        "category": "Constitutional",
        "description": "Fundamental right to life judgments"
    },
    {
        "query": "Dowry death cases under Section 304B IPC",
        "category": "Criminal",
        "description": "Dowry death related judgments"
    },
    {
        "query": "Quashing of FIR under Section 482 CrPC where parties compromised",
        "category": "Procedural",
        "description": "Compoundable offence quashing cases"
    },
    {
        "query": "POCSO Act cases from Supreme Court after 2020",
        "category": "Criminal",
        "description": "Child protection cases"
    },
    {
        "query": "Arbitration award enforcement cases from Delhi High Court",
        "category": "Commercial",
        "description": "Arbitration and Conciliation Act cases"
    },
    {
        "query": "Natural justice principles violation cases",
        "category": "Administrative",
        "description": "Audi alteram partem and fair hearing cases"
    },
]


# Court options for dropdowns
COURTS = [
    {"value": "supreme court", "label": "Supreme Court of India"},
    {"value": "delhi high court", "label": "Delhi High Court"},
    {"value": "bombay high court", "label": "Bombay High Court"},
    {"value": "madras high court", "label": "Madras High Court"},
    {"value": "calcutta high court", "label": "Calcutta High Court"},
    {"value": "karnataka high court", "label": "Karnataka High Court"},
    {"value": "allahabad high court", "label": "Allahabad High Court"},
    {"value": "gujarat high court", "label": "Gujarat High Court"},
    {"value": "punjab haryana high court", "label": "Punjab & Haryana High Court"},
    {"value": "rajasthan high court", "label": "Rajasthan High Court"},
]


# Common sections for dropdowns
COMMON_SECTIONS = {
    "IPC": [
        {"value": "302", "label": "302 - Murder"},
        {"value": "304", "label": "304 - Culpable Homicide"},
        {"value": "304B", "label": "304B - Dowry Death"},
        {"value": "306", "label": "306 - Abetment of Suicide"},
        {"value": "307", "label": "307 - Attempt to Murder"},
        {"value": "376", "label": "376 - Rape"},
        {"value": "420", "label": "420 - Cheating"},
        {"value": "467", "label": "467 - Forgery of Documents"},
        {"value": "468", "label": "468 - Forgery for Cheating"},
        {"value": "471", "label": "471 - Using Forged Documents"},
        {"value": "498A", "label": "498A - Cruelty by Husband"},
        {"value": "406", "label": "406 - Criminal Breach of Trust"},
        {"value": "323", "label": "323 - Voluntarily Causing Hurt"},
        {"value": "354", "label": "354 - Assault/Outraging Modesty"},
        {"value": "506", "label": "506 - Criminal Intimidation"},
    ],
    "CrPC": [
        {"value": "125", "label": "125 - Maintenance"},
        {"value": "144", "label": "144 - Prohibitory Orders"},
        {"value": "161", "label": "161 - Examination of Witnesses"},
        {"value": "164", "label": "164 - Recording of Confessions"},
        {"value": "167", "label": "167 - Remand/Default Bail"},
        {"value": "173", "label": "173 - Charge Sheet"},
        {"value": "227", "label": "227 - Discharge"},
        {"value": "239", "label": "239 - Discharge in Warrant Cases"},
        {"value": "311", "label": "311 - Power to Summon Witnesses"},
        {"value": "319", "label": "319 - Power to Add Accused"},
        {"value": "438", "label": "438 - Anticipatory Bail"},
        {"value": "439", "label": "439 - Regular Bail"},
        {"value": "482", "label": "482 - Inherent Powers/Quashing"},
    ],
    "Constitution": [
        {"value": "14", "label": "Article 14 - Equality"},
        {"value": "19", "label": "Article 19 - Freedoms"},
        {"value": "21", "label": "Article 21 - Right to Life"},
        {"value": "32", "label": "Article 32 - Constitutional Remedies"},
        {"value": "226", "label": "Article 226 - High Court Writs"},
        {"value": "136", "label": "Article 136 - Special Leave Petition"},
        {"value": "142", "label": "Article 142 - Supreme Court Powers"},
    ],
}


class QuerySuggestions:
    """
    Provides intelligent query suggestions based on context and history.
    """

    def __init__(self):
        self.quick_actions = QUICK_ACTIONS
        self.suggested_queries = SUGGESTED_QUERIES
        self.courts = COURTS
        self.sections = COMMON_SECTIONS

    def get_quick_actions(self, category: str = None) -> List[Dict[str, Any]]:
        """Get quick action buttons, optionally filtered by category."""
        actions = self.quick_actions
        if category:
            actions = [a for a in actions if a.category.lower() == category.lower()]

        return [
            {
                "id": a.id,
                "label": a.label,
                "icon": a.icon,
                "query_template": a.query_template,
                "description": a.description,
                "category": a.category
            }
            for a in actions
        ]

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(a.category for a in self.quick_actions))

    def get_suggested_queries(self, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get suggested queries, optionally filtered by category."""
        queries = self.suggested_queries
        if category:
            queries = [q for q in queries if q["category"].lower() == category.lower()]

        return queries[:limit]

    def get_courts(self) -> List[Dict[str, str]]:
        """Get list of courts for dropdown."""
        return self.courts

    def get_sections(self, act: str = None) -> Dict[str, List[Dict[str, str]]]:
        """Get common sections, optionally for a specific act."""
        if act:
            return {act: self.sections.get(act, [])}
        return self.sections

    def build_query(
        self,
        action_id: str,
        params: Dict[str, str]
    ) -> str:
        """
        Build a query from a quick action template.

        Args:
            action_id: ID of the quick action
            params: Parameters to fill in the template

        Returns:
            Formatted query string
        """
        action = next((a for a in self.quick_actions if a.id == action_id), None)
        if not action:
            return ""

        query = action.query_template
        for key, value in params.items():
            query = query.replace(f"{{{key}}}", value)

        # Remove unfilled placeholders
        import re
        query = re.sub(r'\{[^}]+\}', '', query)
        query = ' '.join(query.split())  # Clean up extra spaces

        return query

    def get_follow_up_suggestions(self, query: str, response: str) -> List[str]:
        """
        Generate follow-up query suggestions based on the current query and response.

        Args:
            query: Original user query
            response: Chatbot response

        Returns:
            List of suggested follow-up queries
        """
        suggestions = []

        # Check for common patterns and suggest follow-ups
        query_lower = query.lower()

        if "bail" in query_lower:
            suggestions.extend([
                "What are the conditions typically imposed in such bail cases?",
                "Show me cases where bail was rejected for similar offences",
                "Find Supreme Court guidelines on anticipatory bail"
            ])

        if "420" in query or "cheating" in query_lower:
            suggestions.extend([
                "Show me cases where cheating charges were quashed",
                "Find cases distinguishing between civil and criminal cheating"
            ])

        if "supreme court" in query_lower:
            suggestions.append("Find similar cases from High Courts")
        elif "high court" in query_lower:
            suggestions.append("Find Supreme Court judgments on this topic")

        if "after 2020" in query_lower or "recent" in query_lower:
            suggestions.append("Show me landmark older cases on this topic")

        # Default suggestions if none matched
        if not suggestions:
            suggestions = [
                "Show me similar cases from other courts",
                "Find landmark judgments on this topic",
                "What are the key legal principles from these cases?"
            ]

        return suggestions[:3]
