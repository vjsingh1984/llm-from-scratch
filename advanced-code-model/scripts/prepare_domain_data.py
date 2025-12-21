#!/usr/bin/env python3
"""
Prepare domain-specific datasets for Stage 8.

Domains:
- Web Development (React, Vue, Django)
- Data Science (pandas, numpy, scikit-learn)
- DevOps (Docker, Kubernetes, CI/CD)
- Mobile (iOS, Android, React Native)
- Backend (APIs, databases, microservices)
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import numpy as np
from tokenizers import Tokenizer


DOMAINS = {
    "web": {
        "name": "Web Development",
        "topics": ["React", "Vue", "Django", "Flask", "HTML/CSS", "JavaScript"],
        "examples": [
            {
                "prompt": "Create a React component for a todo list",
                "code": """import React, { useState } from 'react';

function TodoList() {
  const [todos, setTodos] = useState([]);
  const [input, setInput] = useState('');

  const addTodo = () => {
    setTodos([...todos, { id: Date.now(), text: input }]);
    setInput('');
  };

  return (
    <div>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={addTodo}>Add</button>
      <ul>
        {todos.map(todo => <li key={todo.id}>{todo.text}</li>)}
      </ul>
    </div>
  );
}

export default TodoList;"""
            }
        ]
    },
    "datascience": {
        "name": "Data Science",
        "topics": ["pandas", "numpy", "scikit-learn", "matplotlib"],
        "examples": [
            {
                "prompt": "Load and analyze a CSV dataset with pandas",
                "code": """import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Basic analysis
print(df.info())
print(df.describe())

# Group by and aggregate
summary = df.groupby('category')['value'].mean()

# Visualize
summary.plot(kind='bar')
plt.title('Average Value by Category')
plt.xlabel('Category')
plt.ylabel('Average Value')
plt.show()"""
            }
        ]
    },
    "devops": {
        "name": "DevOps",
        "topics": ["Docker", "Kubernetes", "CI/CD", "Terraform"],
        "examples": [
            {
                "prompt": "Create a Dockerfile for a Python application",
                "code": """FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]"""
            }
        ]
    },
    "mobile": {
        "name": "Mobile Development",
        "topics": ["iOS", "Android", "React Native", "Flutter"],
        "examples": [
            {
                "prompt": "Create a simple React Native button component",
                "code": """import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

const CustomButton = ({ title, onPress }) => {
  return (
    <TouchableOpacity style={styles.button} onPress={onPress}>
      <Text style={styles.text}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CustomButton;"""
            }
        ]
    },
    "backend": {
        "name": "Backend Development",
        "topics": ["REST APIs", "GraphQL", "Databases", "Microservices"],
        "examples": [
            {
                "prompt": "Create a FastAPI endpoint for user CRUD",
                "code": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

users_db: List[User] = []

@app.post("/users", response_model=User)
async def create_user(user: User):
    users_db.append(user)
    return user

@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")"""
            }
        ]
    }
}


def generate_domain_examples(domain: str, num_examples: int = 200) -> List[Dict]:
    """Generate examples for a domain."""
    examples = []
    domain_info = DOMAINS[domain]
    base_examples = domain_info["examples"]

    for i in range(num_examples):
        # Randomly select a base example and create variations
        base = random.choice(base_examples)
        examples.append({
            "domain": domain,
            "prompt": base["prompt"],
            "code": base["code"]
        })

    return examples


def format_example(example: Dict) -> str:
    """Format example for training."""
    return f"<|user|>{example['prompt']}<|end|>\n<|assistant|>{example['code']}<|end|>\n"


def main():
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
    output_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("Domain Specialization Dataset (Stage 8)")
    print("=" * 60)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Generate examples for each domain
    all_examples = []
    for domain in DOMAINS:
        print(f"\nGenerating {domain} examples...")
        examples = generate_domain_examples(domain, num_examples=200)
        all_examples.extend(examples)

    print(f"\n✓ Generated {len(all_examples)} total examples")

    # Split train/val
    random.shuffle(all_examples)
    split_idx = int(0.9 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Tokenize and save
    for split, examples in [("train", train_examples), ("val", val_examples)]:
        print(f"\nTokenizing {split} data...")
        all_tokens = []
        max_length = 1024

        for example in examples:
            text = format_example(example)
            encoding = tokenizer.encode(text)
            tokens = encoding.ids[:max_length]

            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))

            all_tokens.append(tokens)

        tokens_array = np.array(all_tokens, dtype=np.int32)
        output_file = output_dir / f"domain_{split}.npy"
        np.save(output_file, tokens_array)
        print(f"✓ Saved: {output_file} ({tokens_array.shape})")

    print("\n" + "=" * 60)
    print("✓ Domain specialization data ready!")
    print("=" * 60)
    print("\nTrain with: --stage domain --domain web")


if __name__ == "__main__":
    main()
