# Warehouse hazard triage from images: d1 decision models vs generative VLMs vs a frontier model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-warehouse-hazards-d1-vlm/notebook.ipynb)

## Question

For spotting safety hazards in warehouse images, how do LiquidAI's new **d1** decision models compare with generative VLMs (including d1-3B's own base model, LFM2.5-VL-3B), a small fine-tuned VLM and a frontier model (Claude Opus 5.5)? For each: how accurate is it, how many hazards does it miss, how many false alarms does it raise, and what does it cost per 1,000 images?

## Dataset

[`Jsohal174/warehouse-safety-hazard-dataset`](https://huggingface.co/datasets/Jsohal174/warehouse-safety-hazard-dataset), licensed **CC BY 4.0**. It has 913 overhead warehouse images in 5 classes: spill, forklift violation, improper stacking, obstacle and safe.

> The images are **synthetic**: Blender renders made photorealistic, with hazards added using Google Gemini. Results describe this benchmark, not real camera footage.

| | Images | Used for |
|---|---|---|
| Train | 627 | Fine-tuning only |
| Calibration | 100 (stratified from the official train split, seed 42) | Choosing the "needs attention" threshold |
| Test | 186 (official test split: 138 hazard, 48 safe) | All reported numbers |

## Models

Every model gets the same 5-way question and class descriptions, on the same images resized to at most 768 px.

| Model | Params | How it answers |
|---|---|---|
| d1-3B, d1-omni-600M | 3.1B, 0.6B | Decision model, zero-shot: class probabilities in one forward pass |
| LFM2.5-VL-3B, Qwen3.5-2B, Qwen3.5-4B | 3B, 2B, 4B | Generative VLM, zero-shot: options A–E, scored by next-token letter probabilities |
| Qwen3.5-0.8B + Unsloth | 0.8B | Vision LoRA fine-tune on the 627 training images, scored the same way |
| Claude Opus 5.5 (optional) | — | Anthropic API with a structured-output label. Paid; runs only with an API key |

The fine-tuned arm uses Unsloth's standard vision fine-tune (`FastVisionModel`), because Unsloth 2026.10.3's decision-model trainer doesn't train on images yet.

## Metrics

- 5-way accuracy (strict, and lenient using the dataset's `accept_also` labels), macro-F1 and per-class recall
- **Needs attention** (any hazard vs safe): hazard recall, false-alarm rate and precision, at P(hazard) ≥ 0.5 and at a threshold chosen on the calibration split for 90% hazard recall
- Calibration of P(hazard), latency, peak GPU memory and **cost per 1,000 images**. Local cost assumes one image at a time on a T4 at $0.35/hour; Claude's uses measured tokens at list price.
- Exact McNemar tests, including d1-3B vs its own base model, LFM2.5-VL-3B

## How to run

1. Click **Open in Colab** above, then choose **Runtime → Change runtime type → T4 GPU**.
2. Optional: to include Claude, add a Colab secret named `ANTHROPIC_API_KEY` (key icon in the left sidebar). The notebook prints a cost estimate before the first full call. Without a key, the Claude arm is skipped. Note that it sends the test images to Anthropic's API.
3. Optional: set `SMOKE_N = 8` in the Config cell for a quick end-to-end check first. It uses a separate cache and a tiny fine-tune.
4. Choose **Runtime → Run all**. Predictions and the fine-tuned adapter are cached under `outputs/`, so a rerun after a disconnect resumes. Set `USE_DRIVE = True` to keep them on Google Drive. The last cell zips and downloads all results.
5. Commit `figures/`, `results.csv`, `summary.csv`, `thresholds.csv` and `reliability_hazard.csv`. The `outputs/` folder is git-ignored.

## Results

Results here: TODO, link to the Quarto post once published.
