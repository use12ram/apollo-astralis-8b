# Apollo Astralis 8B Usage Guide

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Deployment Methods](#deployment-methods)
3. [Usage Patterns](#usage-patterns)
4. [Advanced Usage](#advanced-usage)
5. [Integration Examples](#integration-examples)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Installation & Setup

### Option 1: Ollama (Recommended)

The simplest way to use Apollo Astralis:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Download the GGUF model file
wget https://huggingface.co/vanta-research/apollo-astralis-8b/resolve/main/apollo_astralis_8b.gguf

# Create Modelfile
cat > Modelfile-apollo-astralis <<EOF
from ./apollo_astralis_8b.gguf

template """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

parameter num_predict 256
parameter temperature 0.7
parameter top_p 0.9
parameter top_k 40
parameter repeat_penalty 1.15
parameter stop <|im_start|>
parameter stop <|im_end|>

system """You are Apollo, a collaborative AI assistant specializing in reasoning and problem-solving. You approach each question with genuine curiosity and enthusiasm, breaking down complex problems into clear steps. When you're uncertain, you think through possibilities openly and invite collaboration. Your goal is to help users understand not just the answer, but the reasoning process itself."""
EOF

# Create the model
ollama create apollo-astralis -f Modelfile-apollo-astralis

# Start chatting!
ollama run apollo-astralis
```

### Option 2: Python with HuggingFace

For programmatic access via Python:

```bash
# Install dependencies
pip install torch transformers peft accelerate

# Or with GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load and apply LoRA adapter
model = PeftModel.from_pretrained(model, "vanta-research/apollo-astralis-8b")
model.eval()

print("Apollo Astralis 8B loaded successfully!")
```

### Option 3: GGUF with llama.cpp

For C++ based deployment:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download model
wget https://huggingface.co/vanta-research/apollo-astralis-8b/resolve/main/apollo_astralis_8b.gguf

# Run inference
./main -m apollo_astralis_8b.gguf \
  --prompt "Solve this problem: If x + 7 = 15, what is x?" \
  --temp 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.15 \
  -n 256
```

## Deployment Methods

### Conservative Mode (Default - 256 tokens)

Best for most tasks with balanced response length:

```dockerfile
# Modelfile
from ./apollo_astralis_8b.gguf

template """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

parameter num_predict 256
parameter temperature 0.7
parameter top_p 0.9
parameter top_k 40
parameter repeat_penalty 1.15
parameter stop <|im_start|>
parameter stop <|im_end|>

system """You are Apollo, a collaborative AI assistant specializing in reasoning and problem-solving. You approach each question with genuine curiosity and enthusiasm, breaking down complex problems into clear steps. When you're uncertain, you think through possibilities openly and invite collaboration. Your goal is to help users understand not just the answer, but the reasoning process itself."""
```

### Unlimited Mode (For Complex Reasoning)

For multi-step reasoning requiring extended chain-of-thought:

```dockerfile
# Modelfile-unlimited
from ./apollo_astralis_8b.gguf

template """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

parameter num_predict -1
parameter temperature 0.7
parameter top_p 0.9
parameter top_k 40
parameter repeat_penalty 1.15
parameter stop <|im_start|>
parameter stop <|im_end|>

system """You are Apollo, a collaborative AI assistant specializing in reasoning and problem-solving. You approach each question with genuine curiosity and enthusiasm, breaking down complex problems into clear steps. When you're uncertain, you think through possibilities openly and invite collaboration. Your goal is to help users understand not just the answer, but the reasoning process itself."""
```

Create with: `ollama create apollo-astralis-unlimited -f Modelfile-unlimited`

## Usage Patterns

### 1. Mathematical Problem Solving

Apollo excels at step-by-step mathematical reasoning:

```python
def solve_math_problem(problem, max_tokens=512):
    """Solve mathematical problems with detailed explanations."""
    prompt = f"Solve this problem step by step: {problem}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
problems = [
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    "Calculate 15% of 240",
    "Solve for x: 3x + 7 = 22",
    "A rectangle has length 8 and width 5. Find its area and perimeter."
]

for problem in problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print(f"{'='*60}")
    solution = solve_math_problem(problem)
    print(solution)
```

**Example Output**:
```
Problem: Solve for x: 3x + 7 = 22

<think>
I need to solve this linear equation step by step:
1. Isolate the term with x
2. Divide to find x
3. Verify the answer
</think>

Let's solve this together!

Step 1: Subtract 7 from both sides
3x + 7 - 7 = 22 - 7
3x = 15

Step 2: Divide both sides by 3
3x ÷ 3 = 15 ÷ 3
x = 5

Step 3: Verify
3(5) + 7 = 15 + 7 = 22 ✓

Therefore, x = 5!
```

### 2. Logical Reasoning

Apollo handles complex logical structures:

```python
def analyze_logic_problem(problem):
    """Analyze logical reasoning problems with clear structure."""
    prompt = f"Analyze this logical reasoning problem: {problem}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
logic_problems = [
    "If all cats are mammals, and Fluffy is a cat, what can we conclude about Fluffy?",
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
    "If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?",
]

for problem in logic_problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print(f"{'='*60}")
    analysis = analyze_logic_problem(problem)
    print(analysis)
```

### 3. Creative Problem Solving

Apollo approaches puzzles with enthusiasm:

```python
def solve_puzzle(puzzle):
    """Solve creative puzzles with step-by-step reasoning."""
    prompt = f"Solve this puzzle: {puzzle}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Classic puzzles
puzzles = [
    "You have a 3-liter jug and a 5-liter jug. How can you measure exactly 4 liters?",
    "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only hold the farmer and one item. How does he get everything across safely?",
    "Three light switches control three light bulbs in another room. You can flip switches but only visit the room once. How do you determine which switch controls which bulb?",
]

for puzzle in puzzles:
    print(f"\n{'='*60}")
    print(f"Puzzle: {puzzle}")
    print(f"{'='*60}")
    solution = solve_puzzle(puzzle)
    print(solution)
```

### 4. Collaborative Brainstorming

Apollo's warm personality shines in collaborative tasks:

```python
def brainstorm_with_apollo(topic, context=""):
    """Brainstorm ideas with Apollo's collaborative approach."""
    prompt = f"Let's brainstorm together about: {topic}"
    if context:
        prompt += f"\n\nContext: {context}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,  # Slightly higher for creativity
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
topics = [
    "How can we make online learning more engaging for students?",
    "What are some creative ways to reduce food waste at home?",
    "How might AI assistants help with mental wellness?",
]

for topic in topics:
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"{'='*60}")
    ideas = brainstorm_with_apollo(topic)
    print(ideas)
```

### 5. Code Reasoning & Debugging

Apollo helps understand and fix code:

```python
def analyze_code(code, question=""):
    """Analyze code with reasoning about logic and improvements."""
    prompt = f"""Analyze this code:

```python
{code}
```

{question if question else "Explain what it does and suggest improvements."}
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([])
print(result)
"""

analysis = analyze_code(buggy_code, "What's wrong with this code and how can we fix it?")
print(analysis)
```

## Advanced Usage

### Batch Processing

Process multiple questions efficiently:

```python
def batch_process(questions, batch_size=4):
    """Process multiple questions in batches."""
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode batch
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results

# Example
questions = [
    "What is 25 + 37?",
    "Explain the concept of recursion",
    "How do I sort a list in Python?",
    "What's the difference between a list and a tuple?",
]

answers = batch_process(questions)
for q, a in zip(questions, answers):
    print(f"\nQ: {q}\nA: {a}\n{'-'*60}")
```

### Memory-Efficient Generation

For limited GPU memory:

```python
def memory_efficient_generate(prompt, max_tokens=400):
    """Generate responses with minimal memory usage."""
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use no_grad and enable KV caching
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            use_cache=True,  # Enable KV caching
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clear cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result
```

### Streaming Responses

For real-time generation:

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_response(prompt):
    """Stream responses token by token."""
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generation in a separate thread
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream output
    print("Apollo: ", end="", flush=True)
    for text in streamer:
        print(text, end="", flush=True)
    print()  # New line at end
    
    thread.join()

# Example
stream_response("Explain quantum entanglement in simple terms")
```

### Custom System Prompts

Adapt Apollo's personality for specific contexts:

```python
def create_custom_prompt(user_message, system_prompt=None):
    """Create a chat prompt with custom system prompt."""
    default_system = """You are Apollo, a collaborative AI assistant specializing in reasoning and problem-solving. You approach each question with genuine curiosity and enthusiasm, breaking down complex problems into clear steps."""
    
    system = system_prompt or default_system
    
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message}
    ]
    
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Example: Focus on brevity
brief_system = """You are Apollo, an AI assistant focused on clear, concise explanations. Provide direct answers with minimal extra commentary, but maintain a friendly tone."""

prompt = create_custom_prompt("What is photosynthesis?", system_prompt=brief_system)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Integration Examples

### FastAPI Server

Deploy Apollo as a REST API:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Apollo Astralis API", version="1.0.0")

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    # Model loading code here...
    print("Apollo Astralis loaded and ready!")

class Question(BaseModel):
    text: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    system_prompt: Optional[str] = None

@app.post("/ask")
async def ask_apollo(question: Question):
    """Ask Apollo a question."""
    try:
        prompt = create_custom_prompt(question.text, question.system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=question.max_tokens,
                temperature=question.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "question": question.text,
            "response": response,
            "model": "apollo-astralis-8b-v5-conservative"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "apollo-astralis-8b"}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Gradio Interface

Create an interactive web UI:

```python
import gradio as gr

def apollo_chat(message, history, temperature=0.7, max_tokens=512):
    """Chat with Apollo using Gradio."""
    # Format conversation history
    chat = []
    for h in history:
        chat.append({"role": "user", "content": h[0]})
        chat.append({"role": "assistant", "content": h[1]})
    chat.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# Create interface
interface = gr.ChatInterface(
    fn=apollo_chat,
    title="Apollo Astralis 8B",
    description="Chat with Apollo - a reasoning-focused AI with warm personality",
    theme=gr.themes.Soft(),
    examples=[
        "Solve for x: 2x + 5 = 17",
        "Explain recursion with a simple example",
        "Help me brainstorm ideas for a science fair project",
        "What's the difference between correlation and causation?",
    ],
    additional_inputs=[
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
        gr.Slider(128, 1024, value=512, step=128, label="Max Tokens"),
    ]
)

interface.launch(share=True)
```

### Command Line Interface

Simple CLI tool:

```python
#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Apollo Astralis CLI")
    parser.add_argument("prompt", help="Question or prompt for Apollo")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    
    args = parser.parse_args()
    
    # Load model if not already loaded
    global model, tokenizer
    if 'model' not in globals():
        print("Loading Apollo Astralis...", file=sys.stderr)
        # Load model...
        print("Ready!", file=sys.stderr)
    
    # Generate response
    if args.stream:
        stream_response(args.prompt)
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

if __name__ == "__main__":
    main()
```

Usage: `./apollo_cli.py "What is the Pythagorean theorem?" --stream`

## Performance Optimization

### GPU Optimization

```python
# Use Flash Attention 2 (if available)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Requires flash-attn package
)

# Use torch.compile for faster inference (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

### CPU Optimization

```python
# Optimize for CPU inference
import torch
torch.set_num_threads(8)  # Adjust based on your CPU

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # Use float32 on CPU
    device_map="cpu"
)

# Use optimized GGUF with llama.cpp instead
```

### Memory Optimization

```python
# 8-bit quantization (requires bitsandbytes)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory error

**Solutions**:
- Use quantized GGUF model (Q4_K_M recommended)
- Reduce `max_new_tokens`
- Use gradient checkpointing
- Enable 8-bit quantization
- Use CPU inference

**Issue**: Slow generation speed

**Solutions**:
- Use GPU instead of CPU
- Enable Flash Attention 2
- Use `torch.compile()`
- Reduce `max_new_tokens`
- Use GGUF with llama.cpp

**Issue**: Responses cut off mid-sentence

**Solutions**:
- Increase `num_predict` parameter (Ollama)
- Increase `max_new_tokens` (Python)
- Use unlimited variant for complex reasoning
- Check for EOS token issues

**Issue**: Extracted answers don't match response content

**Solutions**:
- Parse final answer after `<think>` blocks
- Look for "Therefore," or "Answer:" markers
- Use regex to extract final conclusions
- Manually verify automated scores

## Best Practices

### Prompt Engineering

**Good Prompts**:
- Clear and specific questions
- Provide context when needed
- Request step-by-step explanations
- Ask for verification of results

**Examples**:
```python
# Good
"Solve for x step by step: 3x + 7 = 22"

# Better
"Solve for x: 3x + 7 = 22. Show your work and verify the answer."

# Best
"I'm learning algebra. Can you solve for x in this equation: 3x + 7 = 22? Please show each step and explain what you're doing."
```

### Temperature Settings

- **0.1-0.3**: Factual questions, mathematics, logical reasoning
- **0.5-0.7**: General conversation, explanations, problem-solving (default)
- **0.8-1.0**: Creative brainstorming, multiple perspectives

### Token Limits

- **128-256**: Quick answers, simple questions
- **256-512**: Standard explanations, moderate reasoning (default)
- **512-1024**: Complex problems, multi-step reasoning
- **Unlimited (-1)**: Extended chain-of-thought, very complex problems

### Answer Extraction

When parsing Apollo's responses programmatically:

```python
import re

def extract_final_answer(response):
    """Extract final answer from Apollo's response."""
    # Look for explicit answer markers
    patterns = [
        r"Therefore,?\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"(?:The )?final answer is\s*(.+?)(?:\n|$)",
        r"x\s*=\s*([^,\n]+)",  # For algebra
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: last non-empty line after <think> block
    if "</think>" in response:
        post_think = response.split("</think>")[-1]
        lines = [l.strip() for l in post_think.split("\n") if l.strip()]
        if lines:
            return lines[-1]
    
    # Ultimate fallback: last non-empty line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else response
```

### Personality Awareness

Apollo's warm personality is intentional, but may not suit all contexts:

**Appropriate**:
- Educational environments
- Collaborative work
- Learning and exploration
- Friendly assistance

**Less appropriate**:
- Formal academic papers
- Clinical documentation
- Legal or medical contexts requiring neutrality
- High-stakes professional advice

### Verification & Validation

Always verify critical information:

```python
def verify_with_apollo(claim, reasoning):
    """Ask Apollo to verify its own reasoning."""
    prompt = f"""Please verify this reasoning:

Claim: {claim}
Reasoning: {reasoning}

Is this correct? If not, what's wrong?"""
    
    # Generate verification response...
    return verification
```

---

## Additional Resources

- **Model Card**: See MODEL_CARD.md for technical details
- **GitHub**: https://github.com/vanta-research/apollo-astralis-8b
- **HuggingFace**: https://huggingface.co/vanta-research/apollo-astralis-8b
- **Documentation**: https://vanta.ai/models/apollo-astralis-8b
- **Issues**: Report bugs and request features on GitHub

## Community & Support

- **Discord**: Join the VANTA Research community
- **Discussions**: HuggingFace model discussions
- **Email**: research@vanta.ai

---

*Happy reasoning with Apollo Astralis! 🚀*
