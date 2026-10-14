# TinyText

**TinyText** is an open-source framework and family of small, efficient language models built from scratch in PyTorch, focused on fast local execution, consumer hardware efficiency, and accessible model research.

This repository serves as the home for all TinyText model releases, starting with the **Flash** series and expanding with future releases.

---

## Releases & Models

### 1. TinyText Flash (Current Release)
A 20M-parameter causal language model engineered for lightning-fast inference (< 90 MB VRAM, > 85 tokens/sec on CPU):

- **`TinyText-Flash-20M-Base`**: 19.27M-parameter foundation model pretrained on a 400M-token educational mixture.
- **`TinyText-Flash-20M-Instruct`**: Supervised instruction-tuned assistant variant fine-tuned with pretraining text replay to mitigate forgetting.

### 2. Future Releases
- Scaled parameter architectures, extended context windows, and advanced reasoning variants will be released under this repository.

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies (Python 3.10+):

```bash
git clone https://github.com/Snaehath/tinytext.git
cd tinytext
pip install -r requirements.txt
```

### 2. Downloading Model Weights

Download the compact release artifact (`tinytext_flash_20m_instruct_inference.pt`, ~73.55 MB) from the [GitHub Releases](https://github.com/Snaehath/tinytext/releases) page into a local `checkpoints/` folder:

```powershell
New-Item -ItemType Directory -Force -Path checkpoints
Invoke-WebRequest -Uri "https://github.com/Snaehath/tinytext/releases/latest/download/tinytext_flash_20m_instruct_inference.pt" -OutFile "checkpoints/tinytext_flash_20m_instruct_inference.pt"
```

---

## Running Inference

### PowerShell (CUDA GPU)

Using a multiline prompt formatted for `TinyText-Flash-20M-Instruct`:

```powershell
$prompt = @"
System:
You are TinyText, a helpful and concise assistant.

User:
Give 3 creative names for a pet turtle.

Assistant:
"@

python -m tinytext.generate `
    --checkpoint checkpoints/tinytext_flash_20m_instruct_inference.pt `
    --prompt $prompt `
    --device cuda `
    --max_tokens 50 `
    --temperature 0.7
```

### PowerShell (CPU)

TinyText runs out of the box on standard consumer CPUs without requiring a dedicated GPU:

```powershell
$prompt = @"
System:
You are TinyText, a helpful and concise assistant.

User:
Explain what gravity is in one simple sentence.

Assistant:
"@

python -m tinytext.generate `
    --checkpoint checkpoints/tinytext_flash_20m_instruct_inference.pt `
    --prompt $prompt `
    --device cpu `
    --max_tokens 40 `
    --temperature 0.7
```

---

## Hardware Footprint & Performance

Tested on a standard consumer laptop (Intel Core i5-11400H / NVIDIA RTX 3050 Laptop GPU):

| Execution Mode | Time to First Token (TTFT) | Generation Throughput | Peak Memory |
| :--- | :---: | :---: | :---: |
| **CPU (PyTorch FP32)** | ~19 ms | ~86 tokens / sec | ~120 MB RAM |
| **CUDA (NVIDIA GPU)** | ~11 ms | ~78 tokens / sec | **84.4 MB VRAM** |

---

## TinyText Flash Architecture

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **19,274,112** (19.27M) | Trainable parameters (< 20M limit) |
| **Vocabulary** | 8,192 | Byte-level BPE tokenizer (`data/tokenizer/tinytext_bpe_8k.json`) |
| **Context Length** | 512 tokens | Sequence window |
| **Layers** | 8 | Pre-normalized Transformer decoder blocks |
| **Hidden Size** | 384 | Embedding and residual dimension |
| **Attention Query Heads** | 6 | Multi-head dimension = 384 |
| **Attention KV Heads** | 2 | Grouped-Query Attention (3:1 query:KV ratio) |
| **Intermediate Size** | 1,408 | SwiGLU MLP feed-forward dimension |
| **Normalization** | RMSNorm | $\epsilon = 10^{-5}$ |
| **Positional Encoding** | RoPE | Rotary Position Embeddings |
| **Weight Tying** | True | Tied input embedding and output LM head |

---

## Limitations

- **Factual Memory**: Due to the 19M parameter size, the model has limited world-knowledge capacity and may hallucinate specific facts.
- **Complex Reasoning**: Multi-step arithmetic and abstract logic are limited at this scale.
- **Experimental Guardrails**: The model is intended for educational, research, and hobbyist exploration; it does not include safety alignment or RLHF filters.

For in-depth evaluation reports, see [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Datasets & License

This project is licensed under the [MIT License](LICENSE).

Training datasets:
- **Base Pretraining**: FineWeb-Edu, Cosmopedia, TinyStories.
- **Instruction Tuning**: Databricks Dolly 15K, UltraChat 200k.
