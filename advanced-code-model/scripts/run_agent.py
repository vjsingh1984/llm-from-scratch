#!/usr/bin/env python3
"""
Agentic Executor (Stage 7).

Runs multi-step tasks with planning and memory.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))
from src.agent.planner import AgentPlanner, TaskStatus, Memory


class Agent:
    """Autonomous agent with planning and memory."""

    def __init__(self, tool_executor=None):
        """
        Args:
            tool_executor: ToolExecutor instance from Stage 3
        """
        self.planner = AgentPlanner()
        self.memory = Memory()
        self.tool_executor = tool_executor

    def run(self, objective: str) -> str:
        """
        Execute objective with planning.

        Args:
            objective: High-level goal

        Returns:
            Final result
        """
        print("=" * 60)
        print(f"🎯 Objective: {objective}")
        print("=" * 60)

        # Create plan
        print("\n📋 Creating plan...")
        tasks = self.planner.create_plan(objective)

        print(f"\n✓ Plan created: {len(tasks)} tasks")
        for i, task in enumerate(tasks):
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  {i+1}. {task.description}{deps}")

        # Execute tasks
        print("\n" + "=" * 60)
        print("🚀 Executing plan...")
        print("=" * 60)

        while not self.planner.is_complete():
            task = self.planner.get_next_task()

            if task is None:
                print("\n⚠️  No more executable tasks (dependencies not satisfied)")
                break

            print(f"\n▶️  Task {task.id + 1}: {task.description}")
            self.planner.update_task_status(task.id, TaskStatus.IN_PROGRESS)

            # Execute task
            try:
                result = self._execute_task(task)
                self.planner.update_task_status(task.id, TaskStatus.COMPLETED, result=result)
                print(f"✓ Completed")

                # Store in memory
                self.memory.add('task_result', result, {'task_id': task.id})

            except Exception as e:
                error_msg = str(e)
                self.planner.update_task_status(task.id, TaskStatus.FAILED, error=error_msg)
                print(f"✗ Failed: {error_msg}")

        # Summary
        summary = self.planner.get_summary()
        print("\n" + "=" * 60)
        print("📊 Execution Summary")
        print("=" * 60)
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"✓ Completed: {summary['completed']}")
        print(f"✗ Failed: {summary['failed']}")
        print(f"⏸  Pending: {summary['pending']}")

        # Return final result
        final_results = [
            t.result for t in self.planner.tasks
            if t.status == TaskStatus.COMPLETED and t.result
        ]

        return "\n\n".join(final_results) if final_results else "No results"

    def _execute_task(self, task) -> str:
        """Execute a single task."""
        # Use tool if specified
        if task.tool and self.tool_executor:
            result = self.tool_executor.execute_tool(
                task.tool,
                task.arguments or {}
            )
            return result

        # Otherwise, simulate execution
        return f"Executed: {task.description}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Executor (Stage 7)")
    parser.add_argument("--objective", type=str, required=True,
                       help="High-level objective to execute")
    args = parser.parse_args()

    # Create agent (without tool executor for demo)
    agent = Agent(tool_executor=None)

    # Run
    result = agent.run(args.objective)

    print("\n" + "=" * 60)
    print("🎉 Final Result")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
