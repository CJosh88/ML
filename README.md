# ML

Small, reproducible machine-learning experiments: new libraries and models tested against well-known baselines on open datasets. Each experiment is a self-contained Jupyter notebook that runs on Google Colab's free GPU.

## Experiments

| Folder | Question | Open |
|---|---|---|
| [`experiments/2026-10-kumo-tabular`](experiments/2026-10-kumo-tabular) | Does zero-tuning NVIDIA Kumo Tabular beat default XGBoost/LightGBM on OpenML tables, and how does it compare with my hand-tuned 2020 Bank Marketing models? **Yes: 25/25 splits, biggest gains on small data, best PR-AUC on Bank Marketing.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-kumo-tabular/notebook.ipynb) |
| [`experiments/2026-10-warehouse-hazards-d1-vlm`](experiments/2026-10-warehouse-hazards-d1-vlm) | Spotting warehouse safety hazards in images: LiquidAI d1 decision models vs generative VLMs (including d1-3B's own base), a fine-tuned 0.8B VLM and Claude Opus 5.5. Accuracy, missed hazards, false alarms and cost per 1,000 images. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-10-warehouse-hazards-d1-vlm/notebook.ipynb) |

## Layout

```
experiments/
  YYYY-MM-<slug>/
    notebook.ipynb     # runnable top to bottom
    requirements.txt   # pinned versions
    README.md          # question, dataset, how to run, results
    figures/           # charts saved by the notebook
```
