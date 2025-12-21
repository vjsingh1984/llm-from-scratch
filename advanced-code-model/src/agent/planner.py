"""
Agentic Planner (Stage 7).

Enables multi-step planning and task decomposition.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Single task in a plan."""
    id: int
    description: str
    status: TaskStatus
    dependencies: List[int]  # IDs of tasks that must complete first
    tool: Optional[str] = None  # Tool to use (execute_python, search_docs, etc.)
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None


class AgentPlanner:
    """Plans and executes multi-step tasks."""

    def __init__(self):
        self.tasks: List[Task] = []
        self.task_id_counter = 0

    def create_plan(self, objective: str) -> List[Task]:
        """
        Create execution plan from objective.

        Args:
            objective: High-level goal

        Returns:
            List of tasks
        """
        # Simplified planning (in production, use LLM to generate plan)
        tasks = self._decompose_objective(objective)
        self.tasks = tasks
        return tasks

    def _decompose_objective(self, objective: str) -> List[Task]:
        """Decompose objective into tasks."""
        # Example decomposition rules
        if "implement" in objective.lower() or "create" in objective.lower():
            return [
                Task(
                    id=0,
                    description="Design the solution architecture",
                    status=TaskStatus.PENDING,
                    dependencies=[]
                ),
                Task(
                    id=1,
                    description="Write the core implementation",
                    status=TaskStatus.PENDING,
                    dependencies=[0],
                    tool="execute_python"
                ),
                Task(
                    id=2,
                    description="Write tests for the implementation",
                    status=TaskStatus.PENDING,
                    dependencies=[1],
                    tool="run_tests"
                ),
                Task(
                    id=3,
                    description="Document the code",
                    status=TaskStatus.PENDING,
                    dependencies=[1, 2]
                )
            ]
        elif "debug" in objective.lower() or "fix" in objective.lower():
            return [
                Task(
                    id=0,
                    description="Analyze the error",
                    status=TaskStatus.PENDING,
                    dependencies=[],
                    tool="analyze_code"
                ),
                Task(
                    id=1,
                    description="Identify the root cause",
                    status=TaskStatus.PENDING,
                    dependencies=[0],
                    tool="debug_code"
                ),
                Task(
                    id=2,
                    description="Implement the fix",
                    status=TaskStatus.PENDING,
                    dependencies=[1],
                    tool="execute_python"
                ),
                Task(
                    id=3,
                    description="Verify the fix works",
                    status=TaskStatus.PENDING,
                    dependencies=[2],
                    tool="run_tests"
                )
            ]
        else:
            # Generic single-task plan
            return [
                Task(
                    id=0,
                    description=objective,
                    status=TaskStatus.PENDING,
                    dependencies=[]
                )
            ]

    def get_next_task(self) -> Optional[Task]:
        """Get next executable task (dependencies satisfied)."""
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            # Check if dependencies are satisfied
            deps_satisfied = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_satisfied:
                return task

        return None

    def update_task_status(self, task_id: int, status: TaskStatus,
                          result: Optional[str] = None,
                          error: Optional[str] = None):
        """Update task status."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                if error:
                    task.error = error
                break

    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            for task in self.tasks
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get plan execution summary."""
        return {
            'total_tasks': len(self.tasks),
            'completed': sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            'failed': sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            'pending': sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            'in_progress': sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS),
        }


class Memory:
    """Agent memory for storing context across tasks."""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []

    def add(self, entry_type: str, content: str, metadata: Optional[Dict] = None):
        """Add memory entry."""
        entry = {
            'type': entry_type,
            'content': content,
            'metadata': metadata or {}
        }

        self.entries.append(entry)

        # Trim if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def search(self, query: str, entry_type: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search memory (simple substring match for now)."""
        results = []

        for entry in reversed(self.entries):  # Most recent first
            if entry_type and entry['type'] != entry_type:
                continue

            if query.lower() in entry['content'].lower():
                results.append(entry)

            if len(results) >= k:
                break

        return results

    def get_recent(self, n: int = 10, entry_type: Optional[str] = None) -> List[Dict]:
        """Get recent entries."""
        entries = self.entries if not entry_type else [
            e for e in self.entries if e['type'] == entry_type
        ]
        return list(reversed(entries[-n:]))

    def clear(self):
        """Clear all memory."""
        self.entries = []
