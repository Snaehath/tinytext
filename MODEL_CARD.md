# Model Card: TinyText Flash 20M

## 1. Model Overview

**TinyText Flash 20M** is a lightweight, decoder-only causal language model engineered from scratch in PyTorch. The model features strictly **19,274,112 trainable parameters** (< 20M limit), designed specifically for instant local inference on consumer CPUs and low-VRAM edge GPUs (< 90 MB peak inference VRAM).

### Model Lineage & Variants:

- **`TinyText-Flash-20M-Base`**: Pretrained on a 400M-token curated educational corpus (FineWeb-Edu, Cosmopedia, TinyStories).
- **`TinyText-Flash-20M-Instruct`**: Fine-tuned on a curated instruction-following dataset (Databricks Dolly 15K, UltraChat 200k) with an 8% pretraining text replay component to mitigate catastrophic forgetting.

---

## 2. Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Trainable Parameters** | **19,274,112** (19.27M) | Exact non-embedding + tied embedding weights |
| **Vocabulary Size** | 8,192 | Byte-Level Byte-Pair Encoding (BPE) |
| **Max Sequence Length** | 512 | Context window |
| **Transformer Layers (`n_layers`)** | 8 | Pre-normalized causal decoder blocks |
| **Hidden Dimension (`d_model`)** | 384 | Embedding and residual dimension |
| **Attention Query Heads (`n_heads`)** | 6 | Head dimension = 64 |
| **Attention KV Heads (`n_kv_heads`)** | 2 | Grouped-Query Attention (3:1 query:KV ratio) |
| **Feed-Forward Dimension** | 1,408 | SwiGLU MLP intermediate dimension |
| **Normalization** | RMSNorm | Pre-norm with $\epsilon = 10^{-5}$ |
| **Position Embeddings** | RoPE | Rotary Position Embeddings |
| **Weight Tying** | True | `lm_head.weight == embed_tokens.weight` |

---

## 3. Dataset Provenance & Licensing

To maintain strict legal and ethical compliance, the provenance and licensing obligations of all source corpora are explicitly documented:

| Model Variant | Source Dataset | Proportion | Stated License | Model Weights Distribution Term |
| :--- | :--- | :---: | :--- | :--- |
| **Base** | FineWeb-Edu | 65% | ODC-By 1.0 | Open Data Commons Attribution |
| **Base** | Cosmopedia | 30% | Apache 2.0 | Apache 2.0 |
| **Base** | TinyStories | 5% | CDLA-Permissive-2.0 | Permissive attribution |
| **Instruct** | Databricks Dolly 15K | 70% | CC BY-SA 3.0 | **CC BY-SA 3.0 (ShareAlike)** |
| **Instruct** | UltraChat 200k | 30% | MIT | MIT |

### License Notice:
- **Repository Source Code**: Licensed under the **MIT License**.
- **`TinyText-Flash-20M-Base` Weights**: Licensed under **MIT License**.
- **`TinyText-Flash-20M-Instruct` Weights**: In adherence to the ShareAlike provision of Databricks Dolly 15K, the instruction-tuned weights are distributed under the **Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA 3.0)**.

---

## 4. Intended Use & Capabilities

### Suitable Applications:
- Fast local text generation on edge devices and CPUs without network dependencies.
- Simple instruction following, short summaries, text rewriting, and simple factual questions.
- Educational demonstrations of small language model pretraining, GQA, RoPE, and SFT mechanics.
- Research into catastrophic forgetting, replay buffers, and distillation at consumer scale.

### Out-of-Scope & Limitations:
- **Not for High-Stakes Decisions**: Do not use for medical, legal, financial, or safety-critical decisions.
- **Limited Parameter Capacity (19M)**: The model has limited internal factual capacity. It cannot perform multi-step mathematical calculations, complex reasoning, or long-context document analysis.
- **Experimental Safety**: The model does not include RLHF or automated content safety filters. Do not deploy in unmoderated user-facing production environments.

---

## 5. Conversation Serialization Format

The `TinyText-Flash-20M-Instruct` model was trained using the following literal format with Byte-Level BPE:

```
System:
You are TinyText, a helpful and concise assistant. Follow the user's instructions, answer accurately when possible, and acknowledge uncertainty when needed.

User:
{user_query}

Assistant:
```

The model terminates generations with `<|endoftext|>`.

---

## 6. Performance Benchmarks

Measured on a standard laptop (Intel Core i5 CPU / NVIDIA RTX 3050 Laptop GPU):

| Execution Mode | Time to First Token (TTFT) | Generation Throughput | Peak Memory Footprint |
| :--- | :---: | :---: | :---: |
| **CPU (PyTorch FP32)** | 19.46 ms | ~86 tokens / sec | ~120 MB RAM |
| **CUDA (NVIDIA GPU)** | 11.31 ms | ~78 tokens / sec | **84.4 MB VRAM** |

---

## 7. Citation & Attribution

If you use TinyText in your research or project, please cite:

```bibtex
@misc{tinytext2026,
  author = {TinyText Contributors},
  title = {TinyText: Fast, Lightweight Language Models for Edge Inference},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Snaehath/tinytext}}
}
```
