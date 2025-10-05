## Model Card: Apollo Astralis 8B

## Summary
- 8B reasoning model with a warm, collaborative assistant style; built on Qwen/Qwen3-8B via LoRA.
- Demonstrates +36% overall improvement vs Base Qwen 8B in a lightweight standard suite (manual-verified): 93% (13/14).
- VRRE semantic evaluation: 22% automated (answer extraction issue) and 89% manual-verified.
- Conservative fine-tuning (292 examples) preserves reasoning quality; loss reduced 0.91 → 0.39.
- Distributed as PEFT adapters; GGUF Q4_K_M also available for local/Ollama.

## Key Specifications
- Name: Apollo Astralis 8B
- Base Model: Qwen/Qwen3-8B
- Type: Causal LM with LoRA adapters (rank 16, alpha 32, ~67M trainable params)
- Context Window: 40K (inherited from base)
- License: Apache 2.0
- Release: October 2025

## Intended Use
Appropriate
- Reasoning-intensive tasks with step-by-step explanations (math, logic, structured analysis)
- Educational assistance, research support, code reasoning, tutoring

Out of Scope
- Professional legal/medical/financial advice or high-stakes decisions without human oversight
- Contexts requiring strictly formal/neutral tone throughout

## Training Overview
- Methodology: Conservative LoRA on a proven reasoning baseline (V3), layering personality without degrading capability.
- Data: 292 curated examples spanning mathematical/logical reasoning and collaborative tone.
- Hardware/Runtime: 1× RTX 3060 (12GB), ~2 hours, bfloat16.
- Outcome: Stable convergence (0.91 → 0.39) and preserved reasoning behavior.

## Evaluation Highlights
Lightweight benchmark suite (manual-verified; compared to Base Qwen 8B):

| Benchmark | Base Qwen3 8B | Apollo | Δ |
|---|---:|---:|---:|
| MMLU (5) | 40% (2/5) | 100% (5/5) | +60% |
| GSM8K (4) | 75% (3/4) | 100% (4/4) | +25% |
| HellaSwag (2) | 50% (1/2) | 50% (1/2) | 0% |
| ARC (3) | 67% (2/3) | 100% (3/3) | +33% |
| Overall (14) | 57% (8/14) | 93% (13/14) | +36% |

VRRE (VANTA Research Reasoning Evaluation)
- Automated: 22% (2/9) due to answer extraction from <think> blocks rather than final conclusions.
- Manual-verified: 89% (8/9) with clear, step-by-step reasoning and consistent tone.

Note: Models that surface chain-of-thought often require tailored extraction to evaluate accurately.

## Artifacts and Deployment
- Adapters: PEFT (adapter_model.safetensors, adapter_config.json)
- Quantization: GGUF Q4_K_M (~4.7GB) for local and Ollama deployment
- Usage and Modelfiles: see README.md (quick starts, conservative/unlimited variants)
- Programmatic and integrations: see USAGE_GUIDE.md (Transformers + PEFT, FastAPI/Gradio)
- Merging and conversion: see MERGE_GUIDE.md (merge adapters, convert to GGUF)

## Limitations
- Explanatory style can be verbose; downstream systems should extract conclusions post-<think>.
- Optimized for English reasoning; not tuned for highly specialized domains or creative writing.
- Educational intent; verification recommended for critical use cases; maintain human oversight.

## Citation
```bibtex
@misc{apollo-astralis-8b-2025,
  title={Apollo Astralis 8B: Conservative LoRA Fine-tuning for Reasoning and Personality},
  author={VANTA Research},
  year={2025},
  url={https://huggingface.co/vanta-research/apollo-astralis-8b},
  note={Base: Qwen/Qwen3-8B}
}
```

## Contact
- tyler@vantaresearch.xyz
- Hugging Face: vanta-research/apollo-astralis-8b
- GitHub: vanta-research/apollo-astralis-8b

