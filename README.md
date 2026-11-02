# Sponsored Question Answering — Empirical Analysis

**Based on:** *"Sponsored Question Answering"* (Mordo, Tennenholtz & Kurland)

## Research Question

In the Sponsored QA framework, an auction platform dictates a **mathematical fusion requirement** — the sponsored answer must reflect an 80/20 blend of the organic answer and the advertisement, governed by λ = 0.8.

The mathematically ideal target in embedding space is:

**E*_math = λ · E_do + (1−λ) · E_di**

But in practice, fusion is performed by an **LLM generating text**. The central question:

> **How should we prompt an LLM so that the generated text's embedding lands as close as possible to the mathematically prescribed point?**

## Methodology

We test three prompting strategies using a real LLM API (Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API) and measure the **semantic gap** to the ideal vector using sentence embeddings (sentence-transformers/all-MiniLM-L6-v2).

| # | Strategy | Prompt Instruction |
|---|---|---|
| 1 | **Explicit Ratio** | *"Merge keeping exactly 80% organic & 20% ad"* |
| 2 | **Subtle Append** | *"Keep organic intact, append a subtle ad mention"* |
| 3 | **Aggressive Promotion** | *"Rewrite organic to heavily promote the advertiser"* |

### Metrics

- **Semantic Gap (Δ):** `1 − cos(E_text, E*_math)` — distance from the ideal target (lower is better)
- **User Utility (U):** `cos(E_text, E_do)` — how well the fused answer preserves organic content
- **Advertiser Value (A):** `cos(E_text, E_di)` — how well the fused answer represents the ad
- **Baseline:** Theoretical unigram model with Laplace smoothing and cross-entropy similarity (per the paper)

## Key Findings

1. **No prompting strategy achieves zero gap** — text generation is an inherently lossy approximation of mathematical vector interpolation.
2. **Subtle Append** achieves the lowest average semantic gap (Δ ≈ 0.025), closely followed by **Explicit Ratio** (Δ ≈ 0.030).
3. **Aggressive Promotion** severely distorts the λ balance (Δ ≈ 0.076), nearly erasing the User–Advertiser margin.
4. **The User–Advertiser trade-off is non-linear** in embedding space, unlike the linear interpolation assumed by unigram models.

## Setup

### Requirements

- Python 3.x
- numpy, scipy, pandas, huggingface_hub, python-dotenv

### API Token

This notebook uses the HuggingFace Inference API. You need a free HF token:

1. Get a token at https://huggingface.co/settings/tokens
2. Create a `.env` file in the project root:
   ```
   HF_TOKEN=your_token_here
   ```

### Run

Open `SponsoredQA_Analysis.ipynb` in Jupyter and run all cells.

## Author

Lihi Bar-Tal — Technion

