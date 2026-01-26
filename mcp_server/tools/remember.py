"""
Remember tool for Claude Memory Palace MCP server.
"""
from typing import Any, List, Optional

from memory_palace.services import remember


def register_remember(mcp):
    """Register the remember tool with the MCP server."""

    @mcp.tool()
    async def memory_remember(
        instance_id: str,
        memory_type: str,
        content: str,
        subject: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        importance: int = 5,
        project: str = "life",
        source_type: str = "explicit",
        source_context: Optional[str] = None,
        source_session_id: Optional[str] = None,
        supersedes_id: Optional[int] = None,
        auto_link: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Store a new memory in the memory palace.

        PROACTIVE USE - STORE MEMORIES WITHOUT BEING ASKED:
        - When decisions are made, store them (type: design_decision)
        - When bugs are found and fixed, store the solution (type: solution, gotcha)
        - When user preferences are expressed, store them (type: preference)
        - When architectural choices are made, store them (type: architecture)
        - When blockers are encountered, store them (type: blocker)
        - When something works that was tricky, store it (type: workaround, solution)
        - The user shouldn't have to tell you to remember - that's why this system exists
        - If it would be useful in a future session, STORE IT NOW

        AUTO-LINKING: New memories are automatically linked to similar existing memories
        via "relates_to" edges (configurable via auto_link config). For explicit supersession
        or other typed relationships, use supersedes_id or call memory_link separately.

        Args:
            instance_id: Which Claude instance is storing this (e.g., "desktop", "code", "web")
            memory_type: Type of memory (open-ended - use existing types or create new ones like: fact, preference, event, context, insight, relationship, architecture, gotcha, blocker, solution, workaround, design_decision)
            content: The actual memory content
            subject: What/who this memory is about (optional but recommended)
            keywords: List of keywords for searchability
            tags: Freeform organizational tags (separate from keywords)
            importance: 1-10, higher = more important (default 5)
            project: Project this memory belongs to (default "life" for non-project memories)
            source_type: How this memory was created (conversation, explicit, inferred, observation)
            source_context: Snippet of original context
            source_session_id: Link back to conversation session
            supersedes_id: If set, create a 'supersedes' edge to this memory and archive it
            auto_link: Override config to enable/disable similarity-based auto-linking (None = use config)

        Returns:
            Dict with id, subject, embedded status, and links_created (if any edges were auto-created)
        """
        return remember(
            instance_id=instance_id,
            memory_type=memory_type,
            content=content,
            subject=subject,
            keywords=keywords,
            tags=tags,
            importance=importance,
            project=project,
            source_type=source_type,
            source_context=source_context,
            source_session_id=source_session_id,
            supersedes_id=supersedes_id,
            auto_link=auto_link
        )
