"""
Download and prepare bash script dataset.

Sources:
1. Sample bash scripts (common patterns)
2. System scripts (if available)
3. Created examples covering common bash patterns
"""

import json
from pathlib import Path
from typing import List
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_scripts() -> List[str]:
    """
    Create sample bash scripts covering common patterns.

    These scripts demonstrate:
    - Variable assignment and usage
    - Functions
    - Conditionals
    - Loops
    - Command execution
    - File operations
    - String manipulation
    - Arrays
    """

    samples = []

    # 1. Basic hello world
    samples.append("""#!/bin/bash
echo "Hello, World!"
""")

    # 2. Variables
    samples.append("""#!/bin/bash
name="User"
echo "Hello, $name!"
echo "Current directory: $PWD"
""")

    # 3. Functions
    samples.append("""#!/bin/bash
greet() {
    echo "Hello, $1!"
}

greet "Alice"
greet "Bob"
""")

    # 4. Conditionals
    samples.append("""#!/bin/bash
if [ -f "file.txt" ]; then
    echo "File exists"
else
    echo "File not found"
fi
""")

    # 5. Loops - for
    samples.append("""#!/bin/bash
for i in 1 2 3 4 5; do
    echo "Number: $i"
done
""")

    # 6. Loops - while
    samples.append("""#!/bin/bash
count=0
while [ $count -lt 5 ]; do
    echo "Count: $count"
    count=$((count + 1))
done
""")

    # 7. Arrays
    samples.append("""#!/bin/bash
fruits=("apple" "banana" "cherry")
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done
""")

    # 8. Command substitution
    samples.append("""#!/bin/bash
current_date=$(date)
echo "Today is: $current_date"
files=$(ls -1)
echo "Files: $files"
""")

    # 9. File operations
    samples.append("""#!/bin/bash
if [ ! -d "backup" ]; then
    mkdir backup
fi
cp file.txt backup/
echo "Backup created"
""")

    # 10. String manipulation
    samples.append("""#!/bin/bash
text="Hello World"
echo "${text,,}"  # lowercase
echo "${text^^}"  # uppercase
echo "${text:0:5}"  # substring
""")

    # 11. Find and grep
    samples.append("""#!/bin/bash
find . -name "*.txt" -type f
grep -r "pattern" .
ls -la | grep ".log"
""")

    # 12. Error handling
    samples.append("""#!/bin/bash
set -e
command_that_might_fail || {
    echo "Command failed"
    exit 1
}
echo "Success"
""")

    # 13. Read user input
    samples.append("""#!/bin/bash
read -p "Enter your name: " name
echo "Hello, $name!"
""")

    # 14. Case statement
    samples.append("""#!/bin/bash
case $1 in
    start)
        echo "Starting..."
        ;;
    stop)
        echo "Stopping..."
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        ;;
esac
""")

    # 15. Arguments
    samples.append("""#!/bin/bash
echo "Script: $0"
echo "First arg: $1"
echo "All args: $@"
echo "Num args: $#"
""")

    # 16. Pipelines
    samples.append("""#!/bin/bash
cat file.txt | grep "error" | wc -l
ps aux | grep python | awk '{print $2}'
""")

    # 17. Redirections
    samples.append("""#!/bin/bash
echo "Log message" >> log.txt
command 2>&1 | tee output.log
cat < input.txt > output.txt
""")

    # 18. Testing
    samples.append("""#!/bin/bash
if [ -z "$VAR" ]; then
    echo "VAR is empty"
fi

if [ $# -eq 0 ]; then
    echo "No arguments"
fi
""")

    # 19. Math operations
    samples.append("""#!/bin/bash
a=10
b=5
sum=$((a + b))
diff=$((a - b))
prod=$((a * b))
echo "Sum: $sum"
""")

    # 20. Here documents
    samples.append("""#!/bin/bash
cat << EOF
This is a
multi-line
text
EOF
""")

    # 21. Background jobs
    samples.append("""#!/bin/bash
long_command &
echo "Running in background"
wait
echo "Done"
""")

    # 22. Exit codes
    samples.append("""#!/bin/bash
command
if [ $? -eq 0 ]; then
    echo "Success"
else
    echo "Failed"
fi
""")

    # 23. Functions with return
    samples.append("""#!/bin/bash
add() {
    result=$(($1 + $2))
    echo $result
}

sum=$(add 5 3)
echo "Sum: $sum"
""")

    # 24. Associative arrays (bash 4+)
    samples.append("""#!/bin/bash
declare -A config
config[host]="localhost"
config[port]="8080"
echo "${config[host]}:${config[port]}"
""")

    # 25. File tests
    samples.append("""#!/bin/bash
if [ -r "file.txt" ]; then
    echo "Readable"
fi

if [ -w "file.txt" ]; then
    echo "Writable"
fi

if [ -x "script.sh" ]; then
    echo "Executable"
fi
""")

    # 26. String comparison
    samples.append("""#!/bin/bash
str1="hello"
str2="world"

if [ "$str1" = "$str2" ]; then
    echo "Equal"
else
    echo "Not equal"
fi
""")

    # 27. Numeric comparison
    samples.append("""#!/bin/bash
a=10
b=20

if [ $a -lt $b ]; then
    echo "$a is less than $b"
fi
""")

    # 28. Environment variables
    samples.append("""#!/bin/bash
export MY_VAR="value"
echo $PATH
echo $HOME
echo $USER
""")

    # 29. Subshells
    samples.append("""#!/bin/bash
(cd /tmp && ls)
echo "Still in: $PWD"
""")

    # 30. Trap signals
    samples.append("""#!/bin/bash
trap "echo 'Interrupted'; exit" INT TERM
while true; do
    sleep 1
done
""")

    return samples


def save_dataset(scripts: List[str], output_dir: Path):
    """Save scripts to dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as individual files
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    for i, script in enumerate(scripts, 1):
        script_file = scripts_dir / f"script_{i:03d}.sh"
        with open(script_file, 'w') as f:
            f.write(script)

    # Save as combined text file
    combined_file = output_dir / "bash_scripts.txt"
    with open(combined_file, 'w') as f:
        for script in scripts:
            f.write(script)
            f.write("\n\n" + "="*60 + "\n\n")

    # Save as JSON
    json_file = output_dir / "bash_scripts.json"
    with open(json_file, 'w') as f:
        json.dump({
            'scripts': scripts,
            'count': len(scripts)
        }, f, indent=2)

    print(f"Saved {len(scripts)} scripts to {output_dir}")
    print(f"  Individual files: {scripts_dir}")
    print(f"  Combined text: {combined_file}")
    print(f"  JSON format: {json_file}")


def analyze_dataset(scripts: List[str]):
    """Analyze the dataset."""
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') + 1 for s in scripts)
    avg_length = total_chars / len(scripts) if scripts else 0

    print(f"\nDataset Statistics:")
    print(f"  Number of scripts: {len(scripts)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Average script length: {avg_length:.1f} characters")
    print(f"  Shortest script: {min(len(s) for s in scripts)} chars")
    print(f"  Longest script: {max(len(s) for s in scripts)} chars")


def main():
    """Main function."""
    print("="*60)
    print("Bash Script Dataset Preparation")
    print("="*60)
    print()

    # Create sample scripts
    print("Creating sample bash scripts...")
    scripts = create_sample_scripts()

    # Analyze
    analyze_dataset(scripts)

    # Save
    output_dir = Path(__file__).parent.parent / "data"
    save_dataset(scripts, output_dir)

    print("\n" + "="*60)
    print("Dataset ready for training!")
    print("="*60)

    # Test tokenization
    print("\nTesting tokenization on first script...")
    from tokenizer import CodeTokenizer

    tokenizer = CodeTokenizer()
    tokenizer.build_default_vocab()

    sample = scripts[0]
    tokens = tokenizer.encode(sample, add_special_tokens=False)
    print(f"  Script length: {len(sample)} chars")
    print(f"  Token count: {len(tokens)}")
    print(f"  Vocabulary size: {len(tokenizer)}")

    # Show sample
    print(f"\nFirst script:")
    print(sample)


if __name__ == '__main__':
    main()
