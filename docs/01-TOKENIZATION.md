# Tokenization: From Text to Numbers

## Why Tokenization Matters

Neural networks work with numbers, not text. Tokenization is the bridge between human language and machine learning.

## Approaches to Tokenization

### 1. Character-Level
**Pros**: Small vocabulary (~256 for ASCII), handles any text
**Cons**: Very long sequences, hard to learn word meanings

Example: `"hello"` → `['h', 'e', 'l', 'l', 'o']`

### 2. Word-Level
**Pros**: Natural linguistic units, short sequences
**Cons**: Huge vocabulary (100K+ words), can't handle unknown words, ignores morphology

Example: `"running quickly"` → `['running', 'quickly']`

### 3. Subword-Level (Best for LLMs)
**Pros**: Balanced vocabulary size, handles rare/new words, captures morphology
**Cons**: Slightly more complex implementation

Example: `"unhappiness"` → `['un', 'happiness']` or `['un', 'happy', 'ness']`

**We'll implement**: Byte Pair Encoding (BPE), used by GPT-2, GPT-3, and many others.

---

## Byte Pair Encoding (BPE)

### Core Idea
Start with characters, iteratively merge the most frequent pairs.

### Algorithm

#### Training Phase
1. Start with vocabulary = all unique characters in dataset
2. Add special tokens: `<|endoftext|>`, `<|unk|>`, etc.
3. Split all words into characters: `"low"` → `['l', 'o', 'w']`
4. Repeat for N iterations (typically 30K-50K merges):
   - Count all adjacent pairs: `('l', 'o')`, `('o', 'w')`, etc.
   - Find most frequent pair
   - Merge that pair into a single token
   - Update vocabulary
5. Save vocabulary and merge rules

#### Example Training Process

**Initial Text**: `"low low low lower lower newest newest newest newest widest widest widest"`

**Iteration 1**: Most frequent pair = `('e', 's')` (appears 6 times)
- Merge: `"es"` becomes a token
- Vocabulary: `['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', 'es']`

**Iteration 2**: Most frequent pair = `('es', 't')` (appears 6 times)
- Merge: `"est"` becomes a token
- Vocabulary: `[..., 'es', 'est']`

**Iteration 3**: Most frequent pair = `('l', 'o')` (appears 5 times)
- Merge: `"lo"` becomes a token

**And so on...**

After many iterations, vocabulary might include:
`['l', 'o', 'w', 'e', 'r', ..., 'lo', 'low', 'lower', 'est', 'newest', ...]`

#### Encoding Phase (Inference)
Given a new sentence, apply merge rules in the order they were learned:

**Input**: `"lowest"`

1. Start: `['l', 'o', 'w', 'e', 's', 't']`
2. Apply merge `('l', 'o')` → `['lo', 'w', 'e', 's', 't']`
3. Apply merge `('lo', 'w')` → `['low', 'e', 's', 't']`
4. Apply merge `('e', 's')` → `['low', 'es', 't']`
5. Apply merge `('es', 't')` → `['low', 'est']`

**Final tokens**: `['low', 'est']`

Convert to IDs using vocabulary mapping:
`['low', 'est']` → `[245, 378]` (example IDs)

---

## Implementation Plan

### Files We'll Create

```
tokenizer/
├── bpe.py           # Core BPE algorithm
├── vocab.py         # Vocabulary management
└── train_tokenizer.py  # Script to train on dataset
```

### Key Classes

#### 1. `Vocabulary` class
```python
class Vocabulary:
    def __init__(self):
        self.token_to_id = {}  # "hello" → 123
        self.id_to_token = {}  # 123 → "hello"

    def add_token(self, token: str) -> int:
        """Add token and return its ID"""

    def encode_token(self, token: str) -> int:
        """Convert token to ID"""

    def decode_token(self, id: int) -> str:
        """Convert ID to token"""
```

#### 2. `BPETokenizer` class
```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = Vocabulary()
        self.merges = []  # List of (pair, merged_token) rules

    def train(self, texts: list[str]):
        """Learn BPE merges from training data"""

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs"""

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text"""
```

---

## Special Tokens

LLMs need special tokens for control:

- `<|endoftext|>` (or `<|eos|>`): Marks end of document
  - Used to separate different documents in training data
  - Helps model learn document boundaries

- `<|unk|>`: Unknown token (for tokens not in vocabulary)
  - Rare in BPE since you can always fall back to characters

- `<|pad|>`: Padding token (for batching sequences of different lengths)
  - ID usually 0

- `<|bos|>`: Beginning of sequence (some models use this)

**Example Training Data**:
```
"Once upon a time<|endoftext|>In a galaxy far away<|endoftext|>The quick brown fox"
```

---

## Practical Considerations

### 1. Byte-Level BPE
Modern implementations (like GPT-2) use **byte-level BPE**:
- Base vocabulary = 256 bytes (UTF-8)
- Can represent ANY text (any language, emojis, etc.)
- No unknown tokens needed
- We'll implement this variant

### 2. Pre-tokenization
Before BPE, split on whitespace and punctuation:
```
"Hello, world!" → ["Hello", ",", " ", "world", "!"]
```

Apply BPE to each segment independently. This prevents merges across word boundaries in undesirable ways.

### 3. Vocabulary Size
- **Small (8K-16K)**: Fast, but longer sequences
- **Medium (32K)**: Good balance (we'll use this)
- **Large (50K-100K)**: Shorter sequences, slower, used by GPT-3

### 4. Training Data Size
- Need representative sample of target domain
- For general English: 10MB+ of diverse text
- We'll start with TinyStories dataset (~100MB)

---

## Code Walkthrough

Let's implement the core BPE algorithm step by step.

### Step 1: Count Pairs

```python
def get_pair_counts(words: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Count frequency of adjacent token pairs.

    Args:
        words: {"hello": 100, "world": 50} (word -> frequency)

    Returns:
        {("h", "e"): 100, ("e", "l"): 200, ...}
    """
    pairs = {}
    for word, freq in words.items():
        symbols = word.split()  # "h e l l o" → ["h", "e", "l", "l", "o"]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs
```

### Step 2: Merge Pair

```python
def merge_pair(word: str, pair: tuple[str, str], new_token: str) -> str:
    """
    Replace all occurrences of pair with new_token in word.

    Example:
        word = "l o w e r"
        pair = ("l", "o")
        new_token = "lo"
        result = "lo w e r"
    """
    symbols = word.split()
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
            merged.append(new_token)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return ' '.join(merged)
```

### Step 3: Training Loop

```python
def train_bpe(texts: list[str], vocab_size: int) -> tuple[dict, list]:
    """
    Train BPE tokenizer.

    Returns:
        vocabulary: {token: id}
        merges: [(pair1, merged1), (pair2, merged2), ...]
    """
    # 1. Pre-tokenize: split into words
    word_freqs = get_word_frequencies(texts)

    # 2. Initialize with characters
    vocab = set()
    word_splits = {}
    for word in word_freqs:
        word_splits[word] = ' '.join(list(word)) + ' </w>'  # "hello" → "h e l l o </w>"
        vocab.update(word.split())

    # 3. Iteratively merge
    merges = []
    while len(vocab) < vocab_size:
        # Count pairs
        pair_freqs = get_pair_counts({w: word_freqs[w.replace(' ', '').replace('</w>', '')]
                                      for w in word_splits.values()})

        if not pair_freqs:
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)

        # Merge and update
        new_token = ''.join(best_pair)
        merges.append((best_pair, new_token))
        vocab.add(new_token)

        # Update all words
        for word in list(word_splits.keys()):
            word_splits[word] = merge_pair(word_splits[word], best_pair, new_token)

    return vocab, merges
```

---

## Testing Your Tokenizer

### Test 1: Can it encode and decode?
```python
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(["hello world", "hello there", "world peace"])

text = "hello world"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

assert decoded == text, "Tokenizer should be reversible!"
```

### Test 2: Does it handle unknown patterns?
```python
# Train on simple words
tokenizer.train(["cat", "dog", "bird"])

# Test on unseen word
encoded = tokenizer.encode("cattle")  # Should break down to known subwords
```

### Test 3: Vocabulary size
```python
assert len(tokenizer.vocab) <= tokenizer.vocab_size
```

---

## Dataset: TinyStories

We'll use the TinyStories dataset for training:
- Stories written in simple language
- ~100MB of text
- Perfect for learning
- Can train to reasonable perplexity on M1 Max

**Download script** (we'll create this):
```python
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")
train_texts = [story['text'] for story in dataset['train'][:10000]]

with open('data/tinystories_sample.txt', 'w') as f:
    f.write('<|endoftext|>'.join(train_texts))
```

---

## Next Steps

1. **Implement** `tokenizer/vocab.py` (Vocabulary class)
2. **Implement** `tokenizer/bpe.py` (BPETokenizer class)
3. **Download** TinyStories dataset
4. **Train** tokenizer on sample data
5. **Test** encoding/decoding

Once tokenizer works, we'll move to building the transformer architecture!

---

## Checkpoint

After completing this section, you should be able to:
- ✓ Explain why subword tokenization is better than word or character level
- ✓ Describe the BPE algorithm in your own words
- ✓ Implement BPE from scratch
- ✓ Train a tokenizer on sample data
- ✓ Understand vocabulary size trade-offs

**Ready to code?** Let's implement the tokenizer in the next step!
