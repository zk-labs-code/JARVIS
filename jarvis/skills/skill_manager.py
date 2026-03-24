"""Skill learning and management system for JARVIS.

Provides ability to learn new skills from the internet, cache them,
and execute them when needed.
"""

import logging
from typing import Any

from jarvis.actions.web import WebAssistant
from jarvis.core.memory import Memory

logger = logging.getLogger("jarvis.skills")


class SkillManager:
    """Manages JARVIS's learned skills and self-learning capabilities."""

    def __init__(self, memory: Memory, web_assistant: WebAssistant):
        """Initialize the skill manager.

        Args:
            memory: Memory instance for storing learned skills.
            web_assistant: Web assistant for online learning.
        """
        self.memory = memory
        self.web = web_assistant
        self._active_skills: dict[str, dict[str, Any]] = {}
        self._load_cached_skills()

    def _load_cached_skills(self) -> None:
        """Load previously cached skills from memory."""
        try:
            knowledge = self.memory.search_knowledge("skill", category="skills", limit=100)
            for entry in knowledge:
                skill_name = entry.get("topic", "")
                if skill_name:
                    self._active_skills[skill_name] = {
                        "description": entry.get("content", ""),
                        "source": entry.get("source", "cached"),
                        "id": entry.get("id"),
                    }
            logger.info(f"Loaded {len(self._active_skills)} cached skills")
        except Exception as e:
            logger.warning(f"Failed to load cached skills: {e}")

    def learn_skill(self, topic: str) -> str:
        """Learn a new skill from the internet.

        Args:
            topic: Skill or topic to learn about.

        Returns:
            Learning result summary.
        """
        logger.info(f"Learning skill: {topic}")

        # Check if already learned
        existing = self.memory.search_knowledge(topic, category="skills", limit=1)
        if existing:
            self.memory.increment_knowledge_usage(existing[0]["id"])
            return (
                f"I already know about '{topic}':\n"
                f"{existing[0]['content'][:500]}"
            )

        # Search and learn from web
        search_query = f"how to {topic} tutorial guide"
        results = self.web.learn_from_web(search_query)

        if not results.get("success"):
            return f"I couldn't find information about '{topic}'. Please try a more specific query."

        # Store the learned knowledge
        content = results.get("content", "")
        sources = results.get("sources", [])
        source_str = ", ".join(sources[:3]) if sources else "web search"

        knowledge_id = self.memory.store_knowledge(
            category="skills",
            topic=topic,
            content=content,
            source=source_str,
            confidence=0.7,
        )

        if knowledge_id > 0:
            self._active_skills[topic] = {
                "description": content[:200],
                "source": source_str,
                "id": knowledge_id,
            }

        summary = f"I've learned about '{topic}'.\n\n"
        summary += content[:1000]
        if len(content) > 1000:
            summary += "\n\n[Full content stored in memory]"

        return summary

    def get_skill(self, skill_name: str) -> dict[str, Any] | None:
        """Get a specific skill by name.

        Args:
            skill_name: Name of the skill.

        Returns:
            Skill data or None.
        """
        # Check active skills
        if skill_name in self._active_skills:
            return self._active_skills[skill_name]

        # Search in memory
        results = self.memory.search_knowledge(
            skill_name, category="skills", limit=1
        )
        if results:
            self.memory.increment_knowledge_usage(results[0]["id"])
            return {
                "description": results[0].get("content", ""),
                "source": results[0].get("source", ""),
                "id": results[0]["id"],
            }

        return None

    def list_skills(self) -> list[str]:
        """List all known skills.

        Returns:
            List of skill names.
        """
        return list(self._active_skills.keys())

    def search_skills(self, query: str) -> list[dict[str, Any]]:
        """Search through learned skills.

        Args:
            query: Search query.

        Returns:
            List of matching skills.
        """
        results = self.memory.search_knowledge(query, category="skills", limit=10)
        return results

    def forget_skill(self, skill_name: str) -> str:
        """Remove a learned skill.

        Args:
            skill_name: Name of the skill to forget.

        Returns:
            Result message.
        """
        if skill_name in self._active_skills:
            del self._active_skills[skill_name]
            return f"Skill '{skill_name}' forgotten from active memory"
        return f"Skill '{skill_name}' not found"

    def get_skill_for_task(self, task_description: str) -> str:
        """Find relevant skills for a given task.

        First checks cached knowledge, then learns from the web if needed.

        Args:
            task_description: Description of the task.

        Returns:
            Relevant skill information.
        """
        # Search existing knowledge
        results = self.memory.search_knowledge(
            task_description, limit=3
        )
        if results:
            best = results[0]
            self.memory.increment_knowledge_usage(best["id"])
            return (
                f"Found relevant knowledge: {best['topic']}\n"
                f"{best['content'][:500]}"
            )

        # Check command history for similar tasks
        similar = self.memory.get_similar_commands(task_description, limit=3)
        if similar:
            return (
                "I've done similar tasks before:\n"
                + "\n".join(
                    f"  - '{cmd['command']}' -> {cmd['action']} (used {cmd['freq']}x)"
                    for cmd in similar
                )
            )

        # Learn from web
        return self.learn_skill(task_description)

    def get_stats(self) -> dict[str, Any]:
        """Get skill manager statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "active_skills": len(self._active_skills),
            "memory_stats": self.memory.get_stats(),
        }
