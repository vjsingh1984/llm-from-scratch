"""
Base Domain Class

Provides the foundation for all domain-specific training data generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random


@dataclass
class DomainExample:
    """Represents a single training example."""
    prompt: str
    code: str
    domain: str
    subdomain: str = ""
    tags: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_for_training(self, format_type: str = "chat") -> str:
        """Format example for training."""
        if format_type == "chat":
            return f"<|user|>{self.prompt}<|end|>\n<|assistant|>{self.code}<|end|>\n"
        elif format_type == "instruction":
            return f"### Instruction:\n{self.prompt}\n\n### Response:\n{self.code}\n"
        elif format_type == "completion":
            return f"# {self.prompt}\n{self.code}\n"
        else:
            return f"{self.prompt}\n{self.code}"


class BaseDomain(ABC):
    """
    Base class for domain-specific training data generators.

    Each domain should:
    1. Define its name with get_name()
    2. Define subdomains with get_subdomains()
    3. Implement get_examples() to return training examples
    """

    def __init__(self):
        self._examples_cache: Optional[List[DomainExample]] = None

    @abstractmethod
    def get_name(self) -> str:
        """Return the domain name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this domain."""
        pass

    @abstractmethod
    def get_subdomains(self) -> List[str]:
        """Return list of subdomains/topics."""
        pass

    @abstractmethod
    def get_examples(self) -> List[DomainExample]:
        """Return all training examples for this domain."""
        pass

    def get_example_count(self) -> int:
        """Return total number of examples."""
        return len(self.get_examples())

    def get_examples_by_subdomain(self, subdomain: str) -> List[DomainExample]:
        """Filter examples by subdomain."""
        return [ex for ex in self.get_examples() if ex.subdomain == subdomain]

    def get_examples_by_difficulty(self, difficulty: str) -> List[DomainExample]:
        """Filter examples by difficulty level."""
        return [ex for ex in self.get_examples() if ex.difficulty == difficulty]

    def get_random_examples(self, n: int) -> List[DomainExample]:
        """Get n random examples."""
        examples = self.get_examples()
        return random.sample(examples, min(n, len(examples)))

    def generate_variations(self, example: DomainExample, n: int = 5) -> List[DomainExample]:
        """
        Generate variations of an example.
        Override in subclasses for domain-specific variations.
        """
        variations = []
        for i in range(n):
            var = DomainExample(
                prompt=example.prompt,
                code=example.code,
                domain=example.domain,
                subdomain=example.subdomain,
                tags=example.tags.copy(),
                difficulty=example.difficulty,
                metadata={**example.metadata, "variation": i}
            )
            variations.append(var)
        return variations

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this domain's examples."""
        examples = self.get_examples()
        subdomains = {}
        difficulties = {"beginner": 0, "intermediate": 0, "advanced": 0}

        for ex in examples:
            subdomains[ex.subdomain] = subdomains.get(ex.subdomain, 0) + 1
            difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1

        return {
            "domain": self.get_name(),
            "total_examples": len(examples),
            "subdomains": subdomains,
            "difficulties": difficulties,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.get_name()}, examples={self.get_example_count()})"


class TemplateDomain(BaseDomain):
    """
    A domain that generates examples from templates.

    Templates use {placeholder} syntax for variable substitution.
    """

    def __init__(self):
        super().__init__()
        self._templates: List[Dict] = []

    def add_template(self, prompt_template: str, code_template: str,
                     variables: Dict[str, List[str]], subdomain: str = "",
                     difficulty: str = "intermediate", tags: List[str] = None):
        """Add a template for example generation."""
        self._templates.append({
            "prompt": prompt_template,
            "code": code_template,
            "variables": variables,
            "subdomain": subdomain,
            "difficulty": difficulty,
            "tags": tags or []
        })

    def generate_from_templates(self, examples_per_template: int = 10) -> List[DomainExample]:
        """Generate examples from all templates."""
        examples = []

        for template in self._templates:
            for _ in range(examples_per_template):
                # Sample variables
                sampled = {k: random.choice(v) for k, v in template["variables"].items()}

                # Fill templates
                prompt = template["prompt"].format(**sampled)
                code = template["code"].format(**sampled)

                examples.append(DomainExample(
                    prompt=prompt,
                    code=code,
                    domain=self.get_name(),
                    subdomain=template["subdomain"],
                    tags=template["tags"],
                    difficulty=template["difficulty"],
                    metadata={"template_vars": sampled}
                ))

        return examples
