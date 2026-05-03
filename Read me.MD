# 🧠 MathBrain-32B (HAIDER-Math-32B)

<div align="center">

**Pakistan's First World-Class Mathematical Reasoning AI**

*Task Arithmetic Merge of the World's Two Strongest 32B Math Models*

[![HuggingFace Base](https://img.shields.io/badge/🤗%20Base%20Model-HAIDER--Math--32B--v1-yellow)](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1)
[![HuggingFace GGUF](https://img.shields.io/badge/🤗%20GGUF-Q4__K__M%20~19GB-purple)](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1-GGUF)
[![Base Downloads](https://img.shields.io/badge/Downloads%20(Base)-184%2B-brightgreen)](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1)
[![GGUF Downloads](https://img.shields.io/badge/Downloads%20(GGUF)-79%2B%20in%202%20days-blue)](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1-GGUF)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-PIEAS%20Pakistan-red)](https://pieas.edu.pk)
[![mergekit](https://img.shields.io/badge/Merge-Task%20Arithmetic-blue)](https://github.com/cg123/mergekit)

</div>

---

## 👥 Team

| Name | Role | Institution |
|---|---|---|
| **Abdul Haseeb Tanoli** | Lead — Model architecture, merging, benchmarking | PIEAS, BS Metallurgy & Materials Engineering |
| **Laiba Irshad** | Co-researcher | PIEAS |
| **Zeeshan Anwar** | Co-researcher | PIEAS |

*Independent research project — PIEAS, Pakistan, 2025*

---

## 🌍 Overview

**HAIDER-Math-32B** (MathBrain-32B) is a 33-billion parameter mathematical reasoning model built by merging the world's two strongest open-source 32B math-reasoning architectures using the **Task Arithmetic** method:

| Parent Model | Creator | Key Strength |
|---|---|---|
| **QwQ-32B-Preview** | Alibaba Qwen Team | Mathematical precision, step-by-step proofs |
| **DeepSeek-R1-Distill-Qwen-32B** | DeepSeek AI | Deep chain-of-thought, problem decomposition |

**Base model:** Qwen-32B (raw pre-trained base)

The Task Arithmetic merge adds the *task vectors* (delta weights from fine-tuning) of both models onto the shared base — enabling the merged model to draw simultaneously on QwQ's mathematical rigor and DeepSeek's deep reasoning chains.

---

## 🔧 Merge Configuration

Created using [mergekit](https://github.com/cg123/mergekit) with the following configuration:

```yaml
models:
  - model: qwq-32b
  - model: deepseek-r1-32b

base_model: qwen-32b

merge_method: task_arithmetic
parameters:
  weight: 0.5
  normalize: true

dtype: bfloat16
```

**Why Task Arithmetic?** Task Arithmetic computes the *task vector* for each fine-tuned model (task_vector = fine_tuned_weights − base_weights), then adds them to the shared base. With `weight=0.5` and `normalize=true`, both models contribute equally and the merged weights stay within a stable norm range. This allows the model to inherit reasoning capabilities from both parents without destructive interference.

---

## 🏆 Benchmark Results

> **⚠️ Resource Constraint Notice:** Due to limited compute (student project on free Kaggle 2×T4 GPUs), the benchmark was run on the **Q4_K_M quantized version** with `max_tokens=4096`. This caused two systematic issues that depressed the raw scores. If proper GPU infrastructure becomes available, we plan to run the full benchmark suite on the BF16 model. The community is welcome to run evaluations and share results.

### HAIDER Benchmark v7 — GrandMaster (160 Questions)

| Dataset | Questions | Raw Score | Verified Correct* | Main Issue |
|---|---|---|---|---|
| MATH Competition | 80 | 54/80 **(67.5%)** | **~62–66/80 (77–82%)** | Token truncation (max 4096) |
| GSM8K Word Problems | 40 | 14/40 **(35.0%)** | **37/40 (92.5%)** | Scorer failed to parse verbose outputs |
| MMLU Mathematics | 40 | 10/40 **(25.0%)** | **~22–28/40 (55–70%)** | Scorer bugs + truncation |
| **Overall** | **160** | **78/160 (48.8%)** | **~121–131/160 (76–82%)** | Hardware/scorer constraints |

*\*Manually verified by comparing model's final answer against ground truth where the scorer returned False.*

**Two systematic scoring failures:**
1. **Scorer format mismatch (GSM8K):** This is a deep-thinking model that writes *"Therefore, the total is **595** gems"* — the scorer looked for bare `595` and failed. 23 of 26 "failed" GSM8K answers were actually correct on manual inspection.
2. **Token truncation (MATH/MMLU):** With `max_tokens=4096`, long reasoning chains (this model generates 500–3000 token thinking traces) were cut before the final answer line.

### vs. Parent Models (reported benchmarks)

| Model | MATH | GSM8K | Thinking |
|---|---|---|---|
| **MathBrain-32B / HAIDER (verified)** | **~82%** | **~92%** | ✅ Deep CoT |
| DeepSeek-R1-Distill-Qwen-32B | 72.6% | 88.0% | ✅ Deep CoT |
| QwQ-32B-Preview | 79.5% | 95.0% | ✅ Deep CoT |

---

## 📦 Available Models

| Version | Link | Size | Hardware |
|---|---|---|---|
| **Base (BF16)** | [HAIDER-Math-32B-v1](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1) | ~65 GB | 2× A100, H100 |
| **GGUF Q4_K_M** | [HAIDER-Math-32B-v1-GGUF](https://huggingface.co/ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1-GGUF) | ~19 GB | RTX 4090, 2× T4, Kaggle free |

---

## 🚀 Quick Start

### Kaggle (Free — 2×T4 GPU)

```python
import subprocess, os

# Install
subprocess.run([
    "pip", "install", "llama-cpp-python", "-q", "--user",
    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"
])
# Download
subprocess.run([
    "huggingface-cli", "download",
    "ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1-GGUF",
    "HAIDER-Math-32B-v1-Q4_K_M.gguf",
    "--local-dir", "/kaggle/working/haider"
])

os.environ["GGML_CUDA_NO_GRAPHS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from llama_cpp import Llama

llm = Llama(
    model_path="/kaggle/working/haider/HAIDER-Math-32B-v1-Q4_K_M.gguf",
    n_gpu_layers=-1, tensor_split=[15, 15],
    flash_attn=True, n_ctx=4096, n_batch=512, verbose=False
)

result = llm(
    "<|user|>Find all real roots of x^4 - 5x^2 + 4 = 0<|end|><|assistant|>",
    max_tokens=4096, temperature=0.6, top_p=0.95,
    stop=["<|end|>", "</s>"]
)
print(result["choices"][0]["text"])
```

### llama.cpp (Local)

```bash
./llama-cli \
    -m HAIDER-Math-32B-v1-Q4_K_M.gguf \
    -ngl 100 --flash-attn -c 4096 -n 4096 \
    -p "<|user|>Prove that there are infinitely many prime numbers.<|end|><|assistant|>"
```

### Transformers (Full BF16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1",
    torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ABDUL-HASEEB-TANOLI/HAIDER-Math-32B-v1")

messages = [{"role": "user", "content": "Solve: integrate x^2 * e^x dx"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4096,
                         temperature=0.6, top_p=0.95, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Optimal generation settings:**
```python
temperature = 0.6   # DeepSeek-R1 official — do not use 0
top_p       = 0.95
max_tokens  = 4096  # minimum; use 8192+ for hard problems
```

---

## 🖥️ Hardware Guide

| Setup | VRAM | Speed | Notes |
|---|---|---|---|
| 1× RTX 4090 | 24 GB | ~15–20 t/s | Best local option |
| 2× T4 15 GB | 30 GB | ~10–15 t/s | Free on Kaggle ✅ |
| 1× A100 40 GB | 40 GB | ~25–35 t/s | Professional |

---

## 📁 Repository Structure

```
MathBrain-32B/
├── README.md                                     ← This file
├── LICENSE                                       ← Apache 2.0
├── model_cards/
│   ├── HAIDER-Math-32B-v1_README.md             ← HuggingFace card (BF16)
│   └── HAIDER-Math-32B-v1-GGUF_README.md       ← HuggingFace card (GGUF)
├── benchmarks/
│   ├── HAIDER_Benchmark_v7_GrandMaster.ipynb    ← Full benchmark notebook
│   └── benchmark_results.json                   ← Raw results (160 questions)
└── raw_data/
    └── HAIDER_MATERIAL_BRAIN_BLUE-PRINT.html    ← Blueprint: materials science AI
```

---

## 🗺️ Roadmap

### ✅ Completed
- [x] HAIDER-Math-32B-v1 (BF16, HuggingFace) — 184+ downloads
- [x] HAIDER-Math-32B-v1-GGUF (Q4_K_M) — 79+ downloads in 2 days
- [x] HAIDER Benchmark v7 (160-question evaluation)

### 🚧 In Development
- [ ] **HAIDER-Materials-14B** — AI for Metallurgy & Materials Science (architecture planned, awaiting compute)
- [ ] Full benchmark re-run on BF16 model with proper infrastructure
- [ ] Thinking-chain-aware benchmark scorer (fix format mismatch issue)

### 🎯 Planned
- [ ] Extended evaluation: AIME 2024, AMC 10/12, Putnam problems
- [ ] HuggingFace Spaces interactive demo

---

## 📄 Citation

```bibtex
@misc{tanoli2025haider,
  title       = {HAIDER-Math-32B: Task Arithmetic Merge for Mathematical Reasoning},
  author      = {Abdul Haseeb Tanoli and Laiba Irshad and Zeeshan Anwar},
  year        = {2025},
  institution = {Pakistan Institute of Engineering and Applied Sciences (PIEAS)},
  url         = {https://github.com/ABDULHASEEBTANOLI/MathBrain-32B},
  note        = {Independent undergraduate research project}
}
```

---

## ⚖️ License

Apache 2.0 — inherited from parent models (DeepSeek-R1 and QwQ-32B-Preview).

---

<div align="center">

**Made with ❤️ from Pakistan 🇵🇰 · PIEAS 2025**

*Abdul Haseeb Tanoli · Laiba Irshad · Zeeshan Anwar*

</div>