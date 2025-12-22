#!/usr/bin/env python3
"""
Example: Use Tool Calling

This script demonstrates how to use the model's tool calling
capabilities to execute functions.

Usage:
    python examples/03_use_tools.py
"""

import sys
from pathlib import Path
import json
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Define available tools
TOOLS = {
    "execute_python": {
        "description": "Execute Python code and return the output",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"}
        }
    },
    "execute_bash": {
        "description": "Execute a bash command and return the output",
        "parameters": {
            "command": {"type": "string", "description": "Bash command to execute"}
        }
    },
    "read_file": {
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file"}
        }
    },
    "list_files": {
        "description": "List files in a directory",
        "parameters": {
            "path": {"type": "string", "description": "Directory path"}
        }
    }
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "execute_python":
        try:
            result = subprocess.run(
                ["python", "-c", arguments["code"]],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout or result.stderr
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "execute_bash":
        try:
            result = subprocess.run(
                arguments["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout or result.stderr
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "read_file":
        try:
            with open(arguments["path"], "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "list_files":
        try:
            path = Path(arguments["path"])
            files = list(path.iterdir())
            return "\n".join(str(f) for f in files[:20])
        except Exception as e:
            return f"Error: {e}"

    else:
        return f"Unknown tool: {tool_name}"


def parse_tool_call(response: str) -> tuple:
    """Parse a tool call from model response.

    Expected format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>
    """
    if "<tool_call>" not in response:
        return None, None

    try:
        start = response.find("<tool_call>") + len("<tool_call>")
        end = response.find("</tool_call>")
        json_str = response[start:end].strip()
        tool_call = json.loads(json_str)
        return tool_call["name"], tool_call.get("arguments", {})
    except Exception:
        return None, None


def main():
    print("=" * 60)
    print("Tool Calling Example")
    print("=" * 60)

    print("\nAvailable Tools:")
    for name, info in TOOLS.items():
        print(f"  - {name}: {info['description']}")

    # Simulate model responses (in real usage, these come from the model)
    example_calls = [
        {
            "prompt": "What's 2 + 2?",
            "response": '<tool_call>\n{"name": "execute_python", "arguments": {"code": "print(2 + 2)"}}\n</tool_call>'
        },
        {
            "prompt": "List files in current directory",
            "response": '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
        },
        {
            "prompt": "What Python version is installed?",
            "response": '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "python --version"}}\n</tool_call>'
        }
    ]

    print("\n" + "=" * 60)
    print("Executing Tool Calls")
    print("=" * 60)

    for i, example in enumerate(example_calls, 1):
        print(f"\n--- Example {i} ---")
        print(f"User: {example['prompt']}")
        print(f"Model response: {example['response'][:50]}...")

        # Parse tool call
        tool_name, arguments = parse_tool_call(example['response'])

        if tool_name:
            print(f"\nTool: {tool_name}")
            print(f"Arguments: {arguments}")

            # Execute
            result = execute_tool(tool_name, arguments)
            print(f"\nResult:\n{result}")
        else:
            print("No tool call detected")

        print("-" * 40)

    print("\n" + "=" * 60)
    print("Tool Calling Demo Complete!")
    print("=" * 60)
    print("\nTo train a model with tool calling:")
    print("  1. python scripts/prepare_tool_calling_data.py")
    print("  2. python scripts/train.py --stage tools --checkpoint models/code_model_best.pth")


if __name__ == "__main__":
    main()
