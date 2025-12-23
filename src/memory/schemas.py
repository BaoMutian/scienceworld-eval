"""Data structures for the ReasoningBank memory system."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class MemoryEntry:
    """A single memory entry containing extracted reasoning strategy.

    Each memory entry represents a distilled piece of knowledge
    that can be applied to similar tasks.
    """
    title: str
    description: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            content=data.get("content", ""),
        )


@dataclass
class Memory:
    """A complete memory item containing task context and extracted strategies.

    Stores the full context of a task execution including the trajectory
    and the distilled memory entries for future retrieval.
    """
    memory_id: str
    task_id: str
    task_type: str  # task_name in ScienceWorld
    query: str  # task goal
    trajectory: List[Dict[str, str]]  # action-observation pairs
    is_success: bool
    memory_items: List[MemoryEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # Retrieval statistics
    retrieval_count: int = 0  # Total times this memory was retrieved
    retrieval_success_count: int = 0  # Times task succeeded after retrieval

    @property
    def retrieval_success_rate(self) -> float:
        """Calculate success rate when this memory is retrieved.

        Returns:
            Success rate (0.0-1.0), or 0.0 if never retrieved.
        """
        if self.retrieval_count == 0:
            return 0.0
        return self.retrieval_success_count / self.retrieval_count

    def record_retrieval(self, task_success: bool) -> None:
        """Record a retrieval event for this memory.

        Args:
            task_success: Whether the task succeeded after using this memory.
        """
        self.retrieval_count += 1
        if task_success:
            self.retrieval_success_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "query": self.query,
            "trajectory": self.trajectory,
            "is_success": self.is_success,
            "memory_items": [item.to_dict() for item in self.memory_items],
            "created_at": self.created_at,
            "retrieval_count": self.retrieval_count,
            "retrieval_success_count": self.retrieval_success_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        memory_items = [
            MemoryEntry.from_dict(item)
            for item in data.get("memory_items", [])
        ]
        return cls(
            memory_id=data.get("memory_id", ""),
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", ""),
            query=data.get("query", ""),
            trajectory=data.get("trajectory", []),
            is_success=data.get("is_success", False),
            memory_items=memory_items,
            created_at=data.get("created_at", datetime.now().isoformat()),
            retrieval_count=data.get("retrieval_count", 0),
            retrieval_success_count=data.get("retrieval_success_count", 0),
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique memory ID."""
        return f"mem_{uuid.uuid4().hex[:12]}"


@dataclass
class RetrievedMemory:
    """A memory retrieved from the memory bank with similarity score.

    Used to pass retrieval results to the prompt builder.
    """
    memory: Memory
    similarity: float

    @property
    def memory_id(self) -> str:
        return self.memory.memory_id

    @property
    def query(self) -> str:
        return self.memory.query

    @property
    def is_success(self) -> bool:
        return self.memory.is_success

    @property
    def trajectory(self) -> List[Dict[str, str]]:
        return self.memory.trajectory

    @property
    def memory_items(self) -> List[MemoryEntry]:
        return self.memory.memory_items

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for logging/results."""
        return {
            "memory_id": self.memory_id,
            "similarity": round(self.similarity, 4),
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "is_success": self.is_success,
            "num_items": len(self.memory_items),
        }
