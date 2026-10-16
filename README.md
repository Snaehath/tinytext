# TinyText

**TinyText** is an open-source framework and family of small, efficient language models built from scratch in PyTorch. Designed for ultra-low latency, consumer hardware efficiency, and transparent model architecture, TinyText models run instantly on edge devices, laptops, and local environments without cloud dependencies.

---

## Models & Releases

### 1. TinyText Flash 20M (Active Release — v1.0.0)

A 19.27M-parameter causal language model engineered for instant local inference (< 90 MB VRAM, > 85 tokens/sec on CPU):

- **`TinyText-Flash-20M-Instruct`** *(Recommended)*: Supervised instruction-tuned assistant checkpoint trained on curated instruction pairs (Dolly 15K, UltraChat) with replay buffering. Calibrated with a default `repetition_penalty=1.15` to ensure clean, coherent, non-repetitive answers.
- **`TinyText-Flash-20M-Base`**: 19.27M-parameter foundation model pretrained from scratch on 400M educational tokens (FineWeb-Edu, Cosmopedia, TinyStories).

### 2. TinyText Pro (In Specification)

The next-generation dense foundation model (80M parameters) currently in architectural specification, engineered to expand reasoning and open-ended generation while maintaining single-GPU trainability.

---

## Quickstart

### 1. Installation

Clone the repository and install requirements (Python 3.10+):

```bash
git clone https://github.com/Snaehath/tinytext.git
cd tinytext
pip install -r requirements.txt
```

### 2. Download Release Weights

Download the official release artifact (`tinytext_flash_20m_instruct_inference.pt`, ~73.6 MB) from the [GitHub Releases](https://github.com/Snaehath/tinytext/releases/latest) page into a local `checkpoints/` folder:

#### PowerShell (Windows):
```powershell
New-Item -ItemType Directory -Force -Path checkpoints
Invoke-WebRequest -Uri "https://github.com/Snaehath/tinytext/releases/latest/download/tinytext_flash_20m_instruct_inference.pt" -OutFile "checkpoints/tinytext_flash_20m_instruct_inference.pt"
```

#### Bash (Linux / macOS):
```bash
mkdir -p checkpoints
curl -L -o checkpoints/tinytext_flash_20m_instruct_inference.pt "https://github.com/Snaehath/tinytext/releases/latest/download/tinytext_flash_20m_instruct_inference.pt"
```

---

## Running Inference

### 1. Command-Line Generation (`tinytext.generate`)

Run inference directly on GPU or CPU using the native causal generation engine:

#### CUDA GPU:
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
    --max_tokens 60 `
    --temperature 0.7 `
    --repetition_penalty 1.15
```

#### CPU (No dedicated GPU required):
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
    --temperature 0.0
```

---

### 2. Interactive CLI Chat Tester (`scripts/chat_tinytext.py`)

Launch an interactive terminal REPL for testing and prompt experimentation:

```powershell
# Interactive chat with canonical instruct checkpoint
python scripts/chat_tinytext.py --checkpoint checkpoints/tinytext_flash_20m_instruct_inference.pt

# One-shot prompt via CLI
python scripts/chat_tinytext.py `
    --checkpoint checkpoints/tinytext_flash_20m_instruct_inference.pt `
    --prompt "What is the capital of France?"
```

---

## Reasoning Orchestrator & Bounded Tool Layer

For deterministic tasks requiring zero hallucination (arithmetic, bounded extractions, constrained lists, text transformations), TinyText includes a modular **Reasoning Orchestrator** ([`tinytext/orchestrator.py`](tinytext/orchestrator.py)):

- **Task Router**: Automatically identifies intent (Arithmetic, Extraction, Constrained Lists, Rewriting, Factual, Direct Generation).
- **AST Safe Math**: Evaluates mathematical expressions deterministically via abstract syntax trees without executing arbitrary code.
- **Bounded Verifier**: Validates list counts, schema conformity, character constraints, and uniqueness with at most one corrective retry.
- **Multi-Turn Conversation Manager**: Manages session state and short-term dialogue context without context-window overflow.

To run generation with orchestrator routing enabled:

```powershell
python -m tinytext.generate `
    --checkpoint checkpoints/tinytext_flash_20m_instruct_inference.pt `
    --prompt "Calculate (24 * 5) + 18" `
    --orchestrate
```

---

## Hardware Profile & Performance Benchmarks

Measured on a standard developer laptop (Intel Core i5-11400H / NVIDIA GeForce RTX 3050 Laptop GPU):

| Execution Mode | Time to First Token (TTFT) | Generation Throughput | Peak VRAM / RAM |
| :--- | :---: | :---: | :---: |
| **CUDA GPU (PyTorch FP16)** | **~11 ms** | **~78 tokens / sec** | **84.4 MB VRAM** |
| **CPU (PyTorch FP32)** | **~19 ms** | **~86 tokens / sec** | **~120 MB RAM** |

---

## Model Architecture Specifications

TinyText Flash is a modern decoder-only Transformer built with state-of-the-art architectural components:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Trainable Parameters** | **19,274,112** (19.27M) | Strictly verified parameter budget |
| **Vocabulary Size** | **8,192** tokens | Standalone Byte-Level BPE (`data/tokenizer/tinytext_bpe_8k.json`) |
| **Context Length** | **512** tokens | Sequence window |
| **Layers** | **8** | Pre-normalized Transformer decoder blocks |
| **Hidden Size ($d_{\text{model}}$)** | **384** | Embedding and residual dimension |
| **Attention Query Heads** | **6** | Head dimension = 64 ($6 \times 64 = 384$) |
| **Attention KV Heads (GQA)**| **2** | Grouped-Query Attention (3:1 query:KV ratio for KV-cache savings) |
| **FFN Intermediate Size** | **1,408** | SwiGLU activation feed-forward dimension |
| **Normalization** | **RMSNorm** | $\epsilon = 10^{-5}$ pre-layer normalization |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings ($\theta = 10000.0$) |
| **Weight Tying** | **True** | Tied token embedding and LM prediction head |

---

## Local Web Playground

The repository includes a private local web playground featuring a FastAPI backend and a modern React + Tailwind UI:

```powershell
# 1. Start FastAPI Backend (Port 8000)
uvicorn local_playground.backend.app:app --host 127.0.0.1 --port 8000

# 2. In another terminal, start Frontend (Port 5173)
cd local_playground/frontend
npm run dev
```

Visit `http://localhost:5173` to test the canonical model with real-time token metrics, latency tracking, and decoding controls.

---

## Limitations

- **Factual Memory:** At 19M parameters, the model is designed for edge efficiency, deterministic tooling, and basic dialogue. It has limited capacity for deep world trivia and may hallucinate without tool grounding.
- **Complex Multi-Step Logic:** Heavy multi-variable math or multi-step inductive proofs require delegating to the orchestrator tool layer.
- **Safety Filtering:** TinyText is designed for educational research and developer experimentation; it does not include commercial RLHF safety filters.

---

## License

This project is licensed under the [MIT License](LICENSE).
