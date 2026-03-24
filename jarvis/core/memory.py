"""Memory and learning system for JARVIS.

Provides persistent storage for conversation history, learned skills,
and user preferences using SQLite.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.memory")


class Memory:
    """Persistent memory system using SQLite for storing conversations,
    learned knowledge, and user preferences.
    """

    def __init__(self, db_path: str, max_history: int = 50):
        """Initialize the memory system.

        Args:
            db_path: Path to the SQLite database file.
            max_history: Maximum conversation entries to keep in active memory.
        """
        self.db_path = db_path
        self.max_history = max_history
        self._conversation_history: list[dict[str, str]] = []

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._load_recent_history()
        logger.info(f"Memory system initialized: {db_path}")

    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Conversation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT
                )
            """)

            # Learned knowledge/skills
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # User preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Command history (for learning patterns)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            """)

            # Skill cache (for web-learned skills)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skill_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    code TEXT NOT NULL,
                    language TEXT DEFAULT 'python',
                    source_url TEXT,
                    verified INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used TEXT
                )
            """)

            conn.commit()

    def _load_recent_history(self) -> None:
        """Load recent conversation history from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                    (self.max_history,),
                )
                rows = cursor.fetchall()
                self._conversation_history = [
                    {"role": row[0], "content": row[1]} for row in reversed(rows)
                ]
        except sqlite3.Error as e:
            logger.error(f"Failed to load conversation history: {e}")

    def add_conversation(self, role: str, content: str, context: str | None = None) -> None:
        """Add a conversation entry.

        Args:
            role: Speaker role ('user', 'assistant', 'system').
            content: Message content.
            context: Optional context information.
        """
        timestamp = datetime.now().isoformat()
        entry = {"role": role, "content": content}
        self._conversation_history.append(entry)

        # Trim in-memory history
        if len(self._conversation_history) > self.max_history:
            self._conversation_history = self._conversation_history[-self.max_history:]

        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO conversations (role, content, timestamp, context) VALUES (?, ?, ?, ?)",
                    (role, content, timestamp, context),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save conversation: {e}")

    def get_conversation_history(self, limit: int | None = None) -> list[dict[str, str]]:
        """Get recent conversation history.

        Args:
            limit: Maximum number of entries to return. None for all in memory.

        Returns:
            List of conversation entries.
        """
        if limit is None:
            return list(self._conversation_history)
        return list(self._conversation_history[-limit:])

    def clear_conversation(self) -> None:
        """Clear the in-memory conversation history."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared from memory")

    # --- Knowledge Management ---

    def store_knowledge(
        self,
        category: str,
        topic: str,
        content: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Store a piece of learned knowledge.

        Args:
            category: Knowledge category (e.g., 'coding', 'system', 'general').
            topic: Topic name.
            content: Knowledge content.
            source: Where the knowledge was learned from.
            confidence: Confidence score (0.0 to 1.0).

        Returns:
            The ID of the stored knowledge entry.
        """
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO knowledge (category, topic, content, source, confidence,
                    created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (category, topic, content, source, confidence, now, now),
                )
                conn.commit()
                knowledge_id = cursor.lastrowid or 0
                logger.info(f"Stored knowledge: [{category}] {topic}")
                return knowledge_id
        except sqlite3.Error as e:
            logger.error(f"Failed to store knowledge: {e}")
            return -1

    def search_knowledge(
        self, query: str, category: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search stored knowledge.

        Args:
            query: Search query string.
            category: Optional category filter.
            limit: Maximum results.

        Returns:
            List of matching knowledge entries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if category:
                    cursor = conn.execute(
                        """SELECT * FROM knowledge
                        WHERE category = ? AND (topic LIKE ? OR content LIKE ?)
                        ORDER BY usage_count DESC, confidence DESC LIMIT ?""",
                        (category, f"%{query}%", f"%{query}%", limit),
                    )
                else:
                    cursor = conn.execute(
                        """SELECT * FROM knowledge
                        WHERE topic LIKE ? OR content LIKE ?
                        ORDER BY usage_count DESC, confidence DESC LIMIT ?""",
                        (f"%{query}%", f"%{query}%", limit),
                    )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []

    def increment_knowledge_usage(self, knowledge_id: int) -> None:
        """Increment the usage count for a knowledge entry.

        Args:
            knowledge_id: ID of the knowledge entry.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE knowledge SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), knowledge_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update knowledge usage: {e}")

    # --- Command History ---

    def log_command(
        self, command: str, action: str, success: bool = True, details: str | None = None
    ) -> None:
        """Log a command execution for pattern learning.

        Args:
            command: The voice/text command.
            action: The action performed.
            success: Whether the command succeeded.
            details: Optional additional details.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO command_history (command, action, success, timestamp, details) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (command, action, int(success), datetime.now().isoformat(), details),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log command: {e}")

    def get_similar_commands(self, command: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find similar past commands for learning.

        Args:
            command: Command to search for.
            limit: Maximum results.

        Returns:
            List of similar past commands.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT command, action, success, COUNT(*) as freq
                    FROM command_history
                    WHERE command LIKE ?
                    GROUP BY command, action
                    ORDER BY freq DESC LIMIT ?""",
                    (f"%{command}%", limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to search commands: {e}")
            return []

    # --- User Preferences ---

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference.

        Args:
            key: Preference key.
            value: Preference value (will be JSON serialized).
        """
        try:
            serialized = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO preferences (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
                    (key, serialized, datetime.now().isoformat(),
                     serialized, datetime.now().isoformat()),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to set preference: {e}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference.

        Args:
            key: Preference key.
            default: Default value if not found.

        Returns:
            The preference value or default.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM preferences WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Failed to get preference: {e}")
        return default

    # --- Skill Cache ---

    def cache_skill(
        self,
        skill_name: str,
        code: str,
        description: str | None = None,
        language: str = "python",
        source_url: str | None = None,
    ) -> None:
        """Cache a learned skill.

        Args:
            skill_name: Name of the skill.
            code: Skill code.
            description: Skill description.
            language: Programming language.
            source_url: Where the skill was learned from.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO skill_cache (skill_name, description, code, language,
                    source_url, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(skill_name) DO UPDATE SET
                    code = ?, description = ?, last_used = ?""",
                    (skill_name, description, code, language, source_url,
                     datetime.now().isoformat(), code, description,
                     datetime.now().isoformat()),
                )
                conn.commit()
                logger.info(f"Cached skill: {skill_name}")
        except sqlite3.Error as e:
            logger.error(f"Failed to cache skill: {e}")

    def get_cached_skill(self, skill_name: str) -> dict[str, Any] | None:
        """Retrieve a cached skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Skill data or None.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM skill_cache WHERE skill_name = ?", (skill_name,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get cached skill: {e}")
            return None

    def get_stats(self) -> dict[str, int]:
        """Get memory system statistics.

        Returns:
            Dictionary with count statistics.
        """
        stats = {
            "conversations": 0,
            "knowledge_entries": 0,
            "commands_logged": 0,
            "cached_skills": 0,
            "preferences": 0,
        }
        try:
            with sqlite3.connect(self.db_path) as conn:
                for table, key in [
                    ("conversations", "conversations"),
                    ("knowledge", "knowledge_entries"),
                    ("command_history", "commands_logged"),
                    ("skill_cache", "cached_skills"),
                    ("preferences", "preferences"),
                ]:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                    stats[key] = cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
        return stats
