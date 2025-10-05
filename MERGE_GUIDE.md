# Apollo Astralis 8B - Adapter Merge Guide

## Overview

This guide explains how to use the Apollo Astralis 8B LoRA adapters with the base Qwen3-8B model. You can either:

1. **Use adapters directly** with PEFT (recommended for development)
2. **Merge adapters** into the base model (recommended for production)

## Option 1: Use Adapters with PEFT (Recommended)

The simplest approach - no merging required:

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

# Load and apply LoRA adapters
model = PeftModel.from_pretrained(model, "vanta-research/apollo-astralis-8b")
model.eval()

print("Apollo Astralis 8B ready!")
```

### Advantages
- ✅ Simple and straightforward
- ✅ No extra disk space required
- ✅ Can easily swap between base and fine-tuned models
- ✅ Faster initial loading

### Disadvantages
- ❌ Slightly slower inference (adapter application overhead)
- ❌ Requires PEFT library

## Option 2: Merge Adapters into Base Model

For production deployments requiring maximum inference speed:

### Step 1: Install Dependencies

```bash
pip install torch transformers peft accelerate
```

### Step 2: Merge Script

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def merge_adapters(
    base_model_name="Qwen/Qwen3-8B",
    adapter_model_name="vanta-research/apollo-astralis-8b",
    output_path="./apollo-astralis-8b-merged"
):
    """Merge LoRA adapters into base model."""
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading adapters: {adapter_model_name}")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    
    print("Merging adapters into base model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Merge complete!")
    return model

# Run merge
merged_model = merge_adapters()
```

### Step 3: Use Merged Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "./apollo-astralis-8b-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./apollo-astralis-8b-merged")

# Use normally - no PEFT required!
```

### Advantages
- ✅ Faster inference (no adapter overhead)
- ✅ No PEFT dependency required
- ✅ Easier to quantize and convert to other formats
- ✅ Better for production deployment

### Disadvantages
- ❌ Requires ~16GB disk space for merged model
- ❌ One-time merge process required
- ❌ Cannot easily swap back to base model

## Option 3: Convert to GGUF for Ollama

For efficient local deployment with Ollama:

### Step 1: Merge Adapters (see Option 2)

### Step 2: Convert to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install Python dependencies
pip install -r requirements.txt

# Convert merged model to GGUF FP16
python convert_hf_to_gguf.py ../apollo-astralis-8b-merged/ \
  --outfile apollo-astralis-8b-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M (recommended)
./llama-quantize apollo-astralis-8b-f16.gguf \
    apollo_astralis_8b.gguf Q4_K_M
```

### Step 3: Deploy with Ollama

```bash
# Create Modelfile
cat > Modelfile <<EOF
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

# Create Ollama model
ollama create apollo-astralis -f Modelfile

# Run it!
ollama run apollo-astralis
```

## Memory-Efficient Merge (For Limited RAM)

If you have limited system RAM, use CPU offloading:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def merge_with_offload(
    base_model_name="Qwen/Qwen3-8B",
    adapter_model_name="vanta-research/apollo-astralis-8b",
    output_path="./apollo-astralis-8b-merged",
    max_memory_gb=8
):
    """Merge with CPU offloading for limited RAM."""
    
    # Calculate max memory per device
    max_memory = {
        0: f"{max_memory_gb}GB",  # GPU
        "cpu": "30GB"  # CPU fallback
    }
    
    print("Loading base model with CPU offloading...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="./offload_tmp"
    )
    
    print("Loading adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    
    print("Merging...")
    model = model.merge_and_unload()
    
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Complete!")

# Run with 8GB GPU limit
merge_with_offload(max_memory_gb=8)
```

## Quantization Options

After merging, you can quantize for reduced memory usage:

### 8-bit Quantization (bitsandbytes)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "./apollo-astralis-8b-merged",
    quantization_config=quantization_config,
    device_map="auto"
)

# Model now uses ~8GB instead of ~16GB
```

### GGUF Quantization (llama.cpp)

Available quantization formats:
- **Q4_K_M** (4.7GB) - Recommended balance of size and quality
- **Q5_K_M** (5.7GB) - Higher quality, slightly larger
- **Q8_0** (8.5GB) - Near-original quality
- **Q2_K** (3.4GB) - Smallest, noticeable quality loss

```bash
# Quantize to different formats
./llama-quantize apollo-astralis-8b-f16.gguf apollo_astralis_8b.gguf Q4_K_M
./llama-quantize apollo-astralis-8b-f16.gguf apollo-astralis-8b-Q5_K_M.gguf Q5_K_M
./llama-quantize apollo-astralis-8b-f16.gguf apollo-astralis-8b-Q8_0.gguf Q8_0
```

## Verification After Merge

Test your merged model:

```python
def test_merged_model(model_path):
    """Quick test to verify merged model works correctly."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Test prompt
    test_prompt = "Solve for x: 2x + 5 = 17"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Test Response:")
    print(response)
    
    # Check for Apollo characteristics
    checks = {
        "thinking_blocks": "<think>" in response or "step" in response.lower(),
        "friendly_tone": any(word in response.lower() for word in ["let's", "great", "!"]),
        "mathematical": "x" in response and ("=" in response or "17" in response)
    }
    
    print("\n✅ Verification:")
    for check, passed in checks.items():
        print(f"  {check}: {'✓' if passed else '✗'}")
    
    return all(checks.values())

# Run verification
test_merged_model("./apollo-astralis-8b-merged")
```

## Troubleshooting

### "Out of memory during merge"
**Solution**: Use memory-efficient merge with CPU offloading (see above)

### "Merged model gives different outputs"
**Solution**: Ensure you're using the same generation parameters (temperature, top_p, etc.)

### "Cannot load merged model"
**Solution**: Check PyTorch and Transformers versions match those used for merging

### "GGUF conversion fails"
**Solution**: 
1. Ensure merged model is in HuggingFace format (not PEFT)
2. Update llama.cpp to latest version
3. Check model has proper config.json

## Performance Comparison

| Method | Inference Speed | Memory Usage | Setup Time | Production Ready |
|--------|----------------|--------------|------------|------------------|
| PEFT Adapters | ~90% base speed | ~16GB | Instant | ✓ |
| Merged FP16 | 100% base speed | ~16GB | 5-10 min | ✓✓ |
| Merged + 8-bit | ~85% base speed | ~8GB | 5-10 min | ✓✓ |
| GGUF Q4_K_M | ~95% base speed | ~5GB | 15-20 min | ✓✓✓ |

## Recommended Workflow

**For Development**: Use PEFT adapters directly
**For Production (Python)**: Merge to FP16 or 8-bit
**For Production (Ollama/Local)**: Convert to GGUF Q4_K_M

## Additional Resources

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Transformers Guide**: https://huggingface.co/docs/transformers
- **Ollama**: https://ollama.ai

## Support

If you encounter issues with merging or conversion:
- Check GitHub issues: https://github.com/vanta-research/apollo-astralis-8b/issues
- HuggingFace discussions: https://huggingface.co/vanta-research/apollo-astralis-8b/discussions
- Email: research@vanta.ai

---

*Apollo Astralis 8B - Merge with confidence! 🚀*
