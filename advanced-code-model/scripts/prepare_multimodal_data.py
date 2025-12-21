#!/usr/bin/env python3
"""
Prepare multi-modal dataset for Stage 5.

Creates image-code pairs:
- UI mockups → HTML/CSS code
- Diagrams → Code explanations
- Screenshots → Code descriptions
- Charts → Data visualization code
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tokenizers import Tokenizer
from PIL import Image, ImageDraw, ImageFont
import random


# Image-code task types
IMAGE_CODE_TASKS = {
    "ui_mockup": {
        "description": "Convert UI mockup to HTML/CSS",
        "prompts": [
            "Convert this button design to HTML/CSS",
            "Generate code for this login form",
            "Create HTML for this navigation bar",
            "Write CSS for this card component",
        ]
    },
    "diagram": {
        "description": "Explain diagram with code",
        "prompts": [
            "Explain this architecture diagram",
            "Describe this flowchart in code",
            "Convert this state machine to code",
            "Implement this class diagram",
        ]
    },
    "screenshot": {
        "description": "Describe UI screenshot",
        "prompts": [
            "Describe what this code does",
            "Explain this UI layout",
            "Identify components in this screen",
            "Analyze this interface design",
        ]
    },
    "chart": {
        "description": "Generate visualization code",
        "prompts": [
            "Create matplotlib code for this chart",
            "Generate plotly code for this visualization",
            "Write code to recreate this graph",
            "Implement this data visualization",
        ]
    }
}


def create_synthetic_image(image_type: str, width: int = 224,
                          height: int = 224) -> Image.Image:
    """
    Create synthetic images for training.

    In production, use real images. This creates simple placeholder images.

    Args:
        image_type: Type of image to create
        width: Image width
        height: Image height

    Returns:
        PIL Image
    """
    # Create blank image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    if image_type == "ui_mockup":
        # Draw a simple button mockup
        button_x, button_y = width // 4, height // 2
        button_w, button_h = width // 2, height // 6

        # Button rectangle
        draw.rectangle(
            [button_x, button_y, button_x + button_w, button_y + button_h],
            fill='#4CAF50',
            outline='#2E7D32',
            width=2
        )

        # Button text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()

        text = "Submit"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = button_x + (button_w - text_w) // 2
        text_y = button_y + (button_h - text_h) // 2
        draw.text((text_x, text_y), text, fill='white', font=font)

    elif image_type == "diagram":
        # Draw a simple flowchart
        box_size = 60
        spacing = 40

        # Box 1
        draw.rectangle([width//2 - box_size//2, 40, width//2 + box_size//2, 40 + box_size],
                      outline='black', width=2)
        draw.text((width//2 - 15, 60), "Start", fill='black')

        # Arrow
        draw.line([width//2, 100 + box_size, width//2, 140 + box_size], fill='black', width=2)

        # Box 2
        draw.rectangle([width//2 - box_size//2, 140 + box_size, width//2 + box_size//2, 140 + 2*box_size],
                      outline='black', width=2)
        draw.text((width//2 - 20, 160 + box_size), "Process", fill='black')

    elif image_type == "screenshot":
        # Draw code-like text
        code_lines = [
            "def hello():",
            "    print('Hi')",
            "    return 42"
        ]

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Courier.ttc", 14)
        except:
            font = ImageFont.load_default()

        y = 20
        for line in code_lines:
            draw.text((10, y), line, fill='black', font=font)
            y += 20

    elif image_type == "chart":
        # Draw simple bar chart
        bar_width = 30
        max_height = height - 40
        bars = [0.3, 0.7, 0.5, 0.9]

        x = 30
        for h in bars:
            bar_height = int(max_height * h)
            draw.rectangle(
                [x, height - 20 - bar_height, x + bar_width, height - 20],
                fill='#2196F3',
                outline='black'
            )
            x += bar_width + 20

    return img


def generate_code_response(task_type: str, prompt: str) -> str:
    """Generate code response for image-code pair."""

    templates = {
        "ui_mockup": """<!-- Green Submit Button -->
<button class="submit-btn">Submit</button>

<style>
.submit-btn {
  background-color: #4CAF50;
  color: white;
  padding: 12px 24px;
  border: 2px solid #2E7D32;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
}

.submit-btn:hover {
  background-color: #45a049;
}
</style>""",

        "diagram": """# Simple Sequential Process Flow

```python
def process_flow():
    # Step 1: Start
    data = initialize()

    # Step 2: Process
    result = process_data(data)

    return result
```

The diagram shows a two-step sequential flow:
1. Start → Initialize the process
2. Process → Transform the data
""",

        "screenshot": """This code defines a simple greeting function:

```python
def hello():
    print('Hi')
    return 42
```

- Function name: `hello()`
- Prints 'Hi' to console
- Returns integer 42
- No parameters required
""",

        "chart": """import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D']
values = [30, 70, 50, 90]

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='#2196F3', edgecolor='black')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.ylim(0, 100)
plt.show()
"""
    }

    return templates.get(task_type, "# Code implementation")


def create_multimodal_example(task_type: str) -> Dict[str, Any]:
    """
    Create a single multi-modal training example.

    Args:
        task_type: Type of task

    Returns:
        Example dictionary
    """
    # Random prompt
    prompt = random.choice(IMAGE_CODE_TASKS[task_type]["prompts"])

    # Generate image (in production, use real images)
    image = create_synthetic_image(task_type)

    # Generate code response
    code = generate_code_response(task_type, prompt)

    return {
        "task_type": task_type,
        "prompt": prompt,
        "image": image,
        "code": code
    }


def format_multimodal_example(example: Dict[str, Any]) -> str:
    """
    Format multi-modal example for training.

    Format:
    <|image|><image_placeholder><|end|>
    <|user|>Prompt<|end|>
    <|assistant|>Code response<|end|>
    """
    formatted = "<|image|><image_placeholder><|end|>\n"
    formatted += f"<|user|>{example['prompt']}<|end|>\n"
    formatted += f"<|assistant|>{example['code']}<|end|>\n"

    return formatted


def generate_multimodal_dataset(num_examples: int = 1000) -> List[Dict[str, Any]]:
    """Generate multi-modal training examples."""
    examples = []

    task_types = list(IMAGE_CODE_TASKS.keys())

    for i in range(num_examples):
        task_type = random.choice(task_types)
        example = create_multimodal_example(task_type)
        examples.append(example)

    return examples


def save_images(examples: List[Dict[str, Any]], output_dir: Path):
    """Save images to disk."""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for i, example in enumerate(examples):
        image_path = images_dir / f"image_{i:05d}.png"
        example['image'].save(image_path)
        example['image_path'] = str(image_path)
        # Remove PIL image from dict (not JSON serializable)
        del example['image']

    print(f"\n✓ Saved {len(examples)} images to {images_dir}")


def tokenize_and_save(examples: List[Dict[str, Any]], tokenizer_path: Path,
                     output_dir: Path, split: str):
    """Tokenize text and save."""
    print(f"\nTokenizing {split} data...")
    print(f"  Examples: {len(examples)}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))

    all_tokens = []
    image_paths = []
    max_length = 1024

    for example in examples:
        # Format text
        text = format_multimodal_example(example)

        # Tokenize
        encoding = tokenizer.encode(text)
        tokens = encoding.ids

        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))

        all_tokens.append(tokens)
        image_paths.append(example['image_path'])

    # Save tokens
    tokens_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_dir / f"multimodal_{split}_tokens.npy", tokens_array)

    # Save image paths
    with open(output_dir / f"multimodal_{split}_images.json", 'w') as f:
        json.dump(image_paths, f, indent=2)

    print(f"  Saved tokens: {output_dir / f'multimodal_{split}_tokens.npy'}")
    print(f"  Saved image paths: {output_dir / f'multimodal_{split}_images.json'}")
    print(f"  Shape: {tokens_array.shape}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer"
    output_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("Multi-Modal Dataset Preparation (Stage 5)")
    print("=" * 60)

    # Generate examples
    print("\nGenerating multi-modal examples...")
    all_examples = generate_multimodal_dataset(num_examples=1000)

    print(f"  Total examples: {len(all_examples)}")
    print(f"  Task types: {', '.join(IMAGE_CODE_TASKS.keys())}")

    # Save images
    save_images(all_examples, output_dir)

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"\n  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")

    # Tokenize and save
    tokenize_and_save(train_examples, tokenizer_path, output_dir, "train")
    tokenize_and_save(val_examples, tokenizer_path, output_dir, "val")

    # Save metadata
    metadata = {
        "num_examples": len(all_examples),
        "task_types": IMAGE_CODE_TASKS,
        "image_size": [224, 224],
        "train_size": len(train_examples),
        "val_size": len(val_examples)
    }

    with open(output_dir / "multimodal_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata: {output_dir / 'multimodal_metadata.json'}")

    print("\n" + "=" * 60)
    print("✓ Multi-modal dataset preparation complete!")
    print("=" * 60)
    print("\nNext step: Train Multi-Modal Model")
    print("  python scripts/train.py --stage multimodal --checkpoint models/rlhf_model_best.pth")


if __name__ == "__main__":
    main()
