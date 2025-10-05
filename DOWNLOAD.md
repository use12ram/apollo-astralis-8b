# 📦 Model Files

The model weights and GGUF files are hosted on HuggingFace and Ollama due to GitHub's file size limitations.

## Download Options

### 🤗 HuggingFace (Full Package)
**Recommended for fine-tuning and research**

- **LoRA Adapters**: `adapter_model.safetensors` (167MB)
- **GGUF Quantized**: `apollo_astralis_8b.gguf` (4.7GB)
- **All configs and tokenizer files included**

```bash
# Clone with Git LFS
git clone https://huggingface.co/vanta-research/apollo-astralis-8b

# Or download specific files
wget https://huggingface.co/vanta-research/apollo-astralis-8b/resolve/main/adapter_model.safetensors
wget https://huggingface.co/vanta-research/apollo-astralis-8b/resolve/main/apollo_astralis_8b.gguf
```

### 🦙 Ollama (Instant Use)
**Recommended for local inference**

```bash
# One command to download and run
ollama run vanta-research/apollo-astralis-8b
```

## What's in This Repo?

This GitHub repository contains:
- ✅ Complete documentation (README, MODEL_CARD, USAGE_GUIDE, MERGE_GUIDE)
- ✅ Configuration files (adapter_config.json, config.json, tokenizer configs)
- ✅ Tokenizer files (vocab.json, merges.txt, tokenizer.json)
- ❌ Model weights (too large for GitHub - get from HuggingFace or Ollama)

## Quick Links

- 🤗 **HuggingFace**: https://huggingface.co/vanta-research/apollo-astralis-8b
- 🦙 **Ollama**: https://ollama.com/vanta-research/apollo-astralis-8b
- 📖 **Full Documentation**: See [README.md](./README.md)
