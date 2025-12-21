#!/usr/bin/env python3
"""
Tool-calling inference for Stage 3 model.

Supports:
- Code execution (Python, Bash)
- Documentation search
- Code analysis
- Debugging
- Testing
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
import json
import subprocess
import sys
from typing import Dict, Any, List, Optional
import re

# Import model
sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config


class ToolExecutor:
    """Execute tools for the coding model."""

    def __init__(self, safe_mode: bool = True):
        """
        Args:
            safe_mode: If True, ask for confirmation before executing code
        """
        self.safe_mode = safe_mode

    def execute_python(self, code: str) -> str:
        """Execute Python code safely."""
        if self.safe_mode:
            print(f"\n{'='*60}")
            print("PYTHON CODE TO EXECUTE:")
            print(code)
            print('='*60)
            response = input("Execute this code? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                return "Execution cancelled by user"

        try:
            # Execute in subprocess for safety
            result = subprocess.run(
                ['python3', '-c', code],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return output.strip()
        except subprocess.TimeoutExpired:
            return "Error: Execution timeout (5s limit)"
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_bash(self, command: str) -> str:
        """Execute bash command safely."""
        if self.safe_mode:
            print(f"\n{'='*60}")
            print("BASH COMMAND TO EXECUTE:")
            print(command)
            print('='*60)
            response = input("Execute this command? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                return "Execution cancelled by user"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return output.strip()
        except subprocess.TimeoutExpired:
            return "Error: Execution timeout (5s limit)"
        except Exception as e:
            return f"Error: {str(e)}"

    def search_docs(self, query: str) -> str:
        """Search Python documentation (simulated)."""
        # In production, this would search actual documentation
        # For now, return simulated results
        docs = {
            "list comprehensions": "List comprehensions: [expr for item in iterable if condition]",
            "dictionary": "dict() creates a dictionary. Access with d[key].",
            "lambda": "lambda args: expression - Anonymous functions",
            "decorators": "@decorator_name - Modify function behavior",
            "generators": "yield - Create memory-efficient iterators",
        }

        query_lower = query.lower()
        for key, value in docs.items():
            if key in query_lower:
                return value

        return f"Documentation for '{query}' not found in local cache"

    def analyze_code(self, code: str, analysis_type: str = "syntax") -> str:
        """Analyze code for issues."""
        if analysis_type == "syntax":
            try:
                compile(code, '<string>', 'exec')
                return "✓ No syntax errors detected"
            except SyntaxError as e:
                return f"Syntax error: {e}"

        elif analysis_type == "security":
            dangerous = ["eval(", "exec(", "os.system(", "__import__"]
            issues = [d for d in dangerous if d in code]
            if issues:
                return f"⚠️ Security issues: {', '.join(issues)} detected"
            return "✓ No obvious security issues"

        elif analysis_type == "complexity":
            lines = code.split('\n')
            return f"Code complexity: {len(lines)} lines"

        return "Unknown analysis type"

    def run_tests(self, code: str, test_framework: str = "pytest") -> str:
        """Run tests (simulated)."""
        # In production, this would actually run tests
        if "assert" in code:
            return self.execute_python(code)
        return "No tests found in code"

    def debug_code(self, code: str, error: str) -> str:
        """Debug code and suggest fixes."""
        suggestions = {
            "NameError": "Variable not defined. Check spelling or define the variable first.",
            "TypeError": "Type mismatch. Check if you're using the right data types.",
            "IndexError": "Index out of range. Check list/array bounds.",
            "KeyError": "Key not in dictionary. Use .get() method or check key existence.",
            "AttributeError": "Object doesn't have that attribute. Check object type.",
        }

        for error_type, suggestion in suggestions.items():
            if error_type in error:
                return f"💡 {suggestion}"

        return f"Error analysis: {error}"

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        if tool_name == "execute_python":
            return self.execute_python(arguments.get("code", ""))
        elif tool_name == "execute_bash":
            return self.execute_bash(arguments.get("command", ""))
        elif tool_name == "search_docs":
            return self.search_docs(arguments.get("query", ""))
        elif tool_name == "analyze_code":
            return self.analyze_code(
                arguments.get("code", ""),
                arguments.get("analysis_type", "syntax")
            )
        elif tool_name == "run_tests":
            return self.run_tests(
                arguments.get("code", ""),
                arguments.get("test_framework", "pytest")
            )
        elif tool_name == "debug_code":
            return self.debug_code(
                arguments.get("code", ""),
                arguments.get("error", "")
            )
        else:
            return f"Unknown tool: {tool_name}"


class ToolCallingModel:
    """Model with tool calling capabilities."""

    def __init__(self, model_path: Path, config, device: str = "mps"):
        """
        Args:
            model_path: Path to Stage 3 checkpoint
            config: Model configuration
            device: Device to use
        """
        self.device = device
        self.config = config

        # Load model
        print("Loading tool-calling model...")
        self.model = create_model_from_config(config, architecture="dense", device=device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # Handle compiled model
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {
                key.replace('_orig_mod.', ''): value
                for key, value in state_dict.items()
            }

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✓ Model loaded")

        # Load tokenizer
        tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer" / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("✓ Tokenizer loaded")

        # Tool executor
        self.executor = ToolExecutor(safe_mode=True)

    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model output."""
        # Format: <|tool_call|>Tool: name\nArguments: {...}<|end|>
        tool_pattern = r'<\|tool_call\|>.*?Tool:\s*(\w+).*?Arguments:\s*({[^}]+})'
        match = re.search(tool_pattern, text, re.DOTALL)

        if match:
            tool_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
                return {"tool": tool_name, "arguments": arguments}
            except json.JSONDecodeError:
                return None
        return None

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, max_tool_calls: int = 3) -> str:
        """
        Generate response with tool calling.

        Args:
            prompt: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_tool_calls: Maximum number of tool calls allowed

        Returns:
            Final response after tool execution
        """
        # Format prompt
        formatted_prompt = f"<|user|>{prompt}<|end|>\n<|assistant|>"

        # Encode
        encoding = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.tensor([encoding.ids], device=self.device)

        print(f"\n{'='*60}")
        print(f"User: {prompt}")
        print('='*60)

        tool_call_count = 0
        full_response = ""

        while tool_call_count < max_tool_calls:
            # Generate
            with torch.no_grad():
                output_ids = input_ids.clone()

                for _ in range(max_new_tokens):
                    logits = self.model(output_ids)
                    next_token_logits = logits[0, -1, :] / temperature

                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=1)

                    # Check for end token
                    if next_token.item() == self.tokenizer.token_to_id("<|end|>"):
                        break

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0].tolist())

            # Check for tool call
            tool_call = self.parse_tool_call(generated_text)

            if tool_call:
                tool_call_count += 1
                print(f"\n🔧 Tool Call #{tool_call_count}")
                print(f"  Tool: {tool_call['tool']}")
                print(f"  Arguments: {json.dumps(tool_call['arguments'], indent=2)}")

                # Execute tool
                result = self.executor.execute_tool(
                    tool_call['tool'],
                    tool_call['arguments']
                )

                print(f"\n📤 Tool Result:")
                print(f"  {result}")

                # Add tool result to context
                tool_result = f"<|tool_result|>\nTool: {tool_call['tool']}\nOutput: {result}\n<|end|>\n<|assistant|>"
                result_encoding = self.tokenizer.encode(tool_result)
                result_ids = torch.tensor([result_encoding.ids], device=self.device)
                input_ids = torch.cat([output_ids, result_ids], dim=1)

            else:
                # No more tool calls, extract final response
                final_response_match = re.search(r'<\|assistant\|>([^<]+)', generated_text.split('<|tool_call|>')[0])
                if final_response_match:
                    full_response = final_response_match.group(1).strip()
                break

        print(f"\n{'='*60}")
        print("Assistant Response:")
        print('='*60)
        print(full_response)
        print()

        return full_response


def main():
    """Interactive tool calling demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Tool-calling model inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 3 checkpoint")
    parser.add_argument("--model-size", type=str, default="large",
                        choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    # Load model
    config = get_config(args.model_size)
    config.use_rmsnorm = True  # Match training
    config.use_rope = True

    model = ToolCallingModel(
        Path(args.checkpoint),
        config,
        device=args.device
    )

    print("\n" + "="*60)
    print("Tool-Calling Code Model (Stage 3)")
    print("="*60)
    print("\nAvailable tools:")
    print("  - execute_python: Run Python code")
    print("  - execute_bash: Run bash commands")
    print("  - search_docs: Search documentation")
    print("  - analyze_code: Analyze code for issues")
    print("  - run_tests: Run unit tests")
    print("  - debug_code: Debug and suggest fixes")
    print("\nType 'quit' to exit\n")

    # Interactive loop
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            model.generate(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
