# api/export.py
"""
Export functionality for chat sessions and search results.
Supports Markdown, HTML, and text formats.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from .session import ChatSession, ChatMessage


class ExportManager:
    """
    Handles exporting chat sessions and search results to various formats.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize export manager.

        Args:
            output_dir: Default directory for exports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_session_markdown(
        self,
        session: ChatSession,
        include_metadata: bool = True
    ) -> str:
        """
        Export a chat session to Markdown format.

        Args:
            session: Chat session to export
            include_metadata: Whether to include session metadata

        Returns:
            Markdown formatted string
        """
        lines = []

        # Header
        lines.append(f"# Legal Research Chat - {session.title}")
        lines.append("")

        if include_metadata:
            lines.append("## Session Information")
            lines.append(f"- **Session ID:** {session.session_id}")
            lines.append(f"- **Created:** {self._format_date(session.created_at)}")
            lines.append(f"- **Last Updated:** {self._format_date(session.updated_at)}")
            lines.append(f"- **Messages:** {len(session.messages)}")
            lines.append("")

        lines.append("## Conversation")
        lines.append("")

        for msg in session.messages:
            role_label = "**You:**" if msg.role == "user" else "**Assistant:**"
            lines.append(f"### {role_label}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

            if msg.metadata:
                if msg.metadata.get("tools_used"):
                    lines.append(f"*Tools used: {', '.join(msg.metadata['tools_used'])}*")
                if msg.metadata.get("query_time"):
                    lines.append(f"*Response time: {msg.metadata['query_time']:.2f}s*")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Footer
        lines.append("")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def export_session_html(
        self,
        session: ChatSession,
        include_metadata: bool = True
    ) -> str:
        """
        Export a chat session to HTML format.

        Args:
            session: Chat session to export
            include_metadata: Whether to include session metadata

        Returns:
            HTML formatted string
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Research Chat - {session.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 2px solid #16213e;
            padding-bottom: 10px;
        }}
        .metadata {{
            background: #e8e8e8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
        }}
        .user {{
            background: #e3f2fd;
            border-left: 4px solid #1976d2;
        }}
        .assistant {{
            background: #fff;
            border-left: 4px solid #4caf50;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .role {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .content {{
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .meta-info {{
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Legal Research Chat</h1>
    <h2>{session.title}</h2>
"""

        if include_metadata:
            html += f"""
    <div class="metadata">
        <strong>Session ID:</strong> {session.session_id}<br>
        <strong>Created:</strong> {self._format_date(session.created_at)}<br>
        <strong>Last Updated:</strong> {self._format_date(session.updated_at)}<br>
        <strong>Messages:</strong> {len(session.messages)}
    </div>
"""

        for msg in session.messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_label = "You" if msg.role == "user" else "Legal Research Assistant"

            # Escape HTML in content but preserve line breaks
            content = msg.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            content = content.replace("\n", "<br>")

            meta_info = ""
            if msg.metadata:
                meta_parts = []
                if msg.metadata.get("tools_used"):
                    meta_parts.append(f"Tools: {', '.join(msg.metadata['tools_used'])}")
                if msg.metadata.get("query_time"):
                    meta_parts.append(f"Response time: {msg.metadata['query_time']:.2f}s")
                if meta_parts:
                    meta_info = f'<div class="meta-info">{" | ".join(meta_parts)}</div>'

            html += f"""
    <div class="message {role_class}">
        <div class="role">{role_label}</div>
        <div class="content">{content}</div>
        {meta_info}
    </div>
"""

        html += f"""
    <div class="footer">
        Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
        return html

    def export_session_text(self, session: ChatSession) -> str:
        """
        Export a chat session to plain text format.

        Args:
            session: Chat session to export

        Returns:
            Plain text formatted string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"LEGAL RESEARCH CHAT - {session.title}")
        lines.append("=" * 60)
        lines.append(f"Session ID: {session.session_id}")
        lines.append(f"Created: {self._format_date(session.created_at)}")
        lines.append("")
        lines.append("-" * 60)

        for msg in session.messages:
            role = "USER" if msg.role == "user" else "ASSISTANT"
            lines.append(f"\n[{role}] ({self._format_date(msg.timestamp)})")
            lines.append(msg.content)
            lines.append("-" * 60)

        lines.append(f"\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def export_session_json(self, session: ChatSession) -> str:
        """
        Export a chat session to JSON format.

        Args:
            session: Chat session to export

        Returns:
            JSON formatted string
        """
        return json.dumps(session.to_dict(), indent=2, ensure_ascii=False)

    def export_results_markdown(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        tools_used: List[str] = None
    ) -> str:
        """
        Export search results to Markdown format.

        Args:
            query: Original query
            response: Chatbot response
            sources: List of source documents
            tools_used: List of tools used

        Returns:
            Markdown formatted string
        """
        lines = []

        lines.append("# Legal Research Results")
        lines.append("")
        lines.append(f"**Query:** {query}")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if tools_used:
            lines.append(f"**Search Tools Used:** {', '.join(tools_used)}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Response")
        lines.append("")
        lines.append(response)
        lines.append("")

        if sources:
            lines.append("---")
            lines.append("")
            lines.append("## Sources")
            lines.append("")

            for i, source in enumerate(sources, 1):
                lines.append(f"### {i}. {source.get('title', 'Unknown Case')}")
                if source.get('court'):
                    lines.append(f"- **Court:** {source['court']}")
                if source.get('year'):
                    lines.append(f"- **Year:** {source['year']}")
                if source.get('citation'):
                    lines.append(f"- **Citation:** {source['citation']}")
                if source.get('document_link'):
                    lines.append(f"- **Document:** [{source['document_link']}]({source['document_link']})")
                lines.append("")

        return "\n".join(lines)

    def save_export(
        self,
        content: str,
        filename: str,
        format: str = "md"
    ) -> Path:
        """
        Save exported content to a file.

        Args:
            content: Content to save
            filename: Base filename (without extension)
            format: File format (md, html, txt, json)

        Returns:
            Path to saved file
        """
        extension = {
            "md": ".md",
            "markdown": ".md",
            "html": ".html",
            "txt": ".txt",
            "text": ".txt",
            "json": ".json"
        }.get(format.lower(), ".txt")

        filepath = self.output_dir / f"{filename}{extension}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def _format_date(self, date_str: str) -> str:
        """Format ISO date string to readable format."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except:
            return date_str

    def get_export_formats(self) -> List[Dict[str, str]]:
        """Get list of available export formats."""
        return [
            {"value": "md", "label": "Markdown (.md)"},
            {"value": "html", "label": "HTML (.html)"},
            {"value": "txt", "label": "Plain Text (.txt)"},
            {"value": "json", "label": "JSON (.json)"},
        ]
