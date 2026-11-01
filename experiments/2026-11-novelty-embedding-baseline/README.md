# Embedding distance vs LLM judges for research-idea novelty

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/ML/blob/main/experiments/2026-11-novelty-embedding-baseline/notebook.ipynb)

## Question

On Ai2's new novelty-judge benchmark, how close do established embedding novelty metrics, with no LLM at all, get to an LLM judge that uses Ai2's own prompt, and to the frontier-judge accuracies Ai2 reports?

Ai2's [Old Ideas, Novel Problems](https://arxiv.org/abs/2610.02022) (October 2026) found LLM novelty judges unstable under small prompt changes, and every condition in it is an LLM. Liu & Zhai's [axiomatic benchmark](https://arxiv.org/abs/2604.15145) (2026) found the simplest embedding metric, Yin et al.'s minimum distance, the strongest of ten novelty systems, but on different data. This experiment runs both kinds of score on the same benchmark items. The metrics aren't new; the comparison is.

## Datasets

- **[allenai/novelty-judge-bench](https://huggingface.co/datasets/allenai/novelty-judge-bench)** (Ai2), licensed **ODC-BY**, pinned to revision `fe201e9`. ICLR 2026 submissions whose reviewers explicitly praised (154) or faulted (145) their originality, from the top and bottom rating deciles of their area, plus 154 ideas from a plain Claude Sonnet 4.5 generator. **Human-Only** (154 pairs, 299 pointwise ideas) is the main split. **Human+Generated** is secondary, because its lower-novelty side is LLM-written and distance could pick up style.
- **[pwc-archive/papers-with-abstracts](https://huggingface.co/datasets/pwc-archive/papers-with-abstracts)**, licensed **CC BY-SA 4.0**, pinned to revision `459480a`. About 576k ML papers with abstracts and dates, frozen in July 2025, before the ICLR 2026 deadline. This is the prior-work corpus.

## Conditions

| Condition | What it is |
|---|---|
| 1 Yin | `1 − max cosine` similarity to the corpus (Yin et al., 2023, *PLOS ONE*) |
| 2 kNN density | `1 − mean cosine` over the 10 nearest papers |
| 3 RND | Relative neighbour density (Wang et al., 2025, [arXiv 2503.01508](https://arxiv.org/abs/2503.01508)): the percentile of the idea's local density among its 100 neighbours' densities, each from 50 nearest papers |
| 4 LLM judge | Qwen2.5-3B-Instruct (September 2024, before the ICLR 2026 submissions) on Ai2's reference prompt and its P1 (vague criteria) and P2 (no criteria) ablations, rendered from [allenai/novelty-eval](https://github.com/allenai/novelty-eval) at commit `9b2e69f` |
| 5 Yin + judge | Logistic regression on one embedding model's Yin score and the judge's reference-prompt probability, 5-fold cross-validated; run once per embedding model |

Conditions 1–3 each run on **two embedding models**, both trained before the ICLR 2026 submissions existed: **bge-small-en-v1.5** (2023; 33M parameters, first 256 tokens) and **BGE-M3** (2024; 568M parameters, first 512 tokens), the model Liu & Zhai used. The second tests whether a weak result is just down to a small embedding model.

**Reference points:** the six frontier judges in Ai2's Figure 4 (Claude Sonnet 4.5, Opus 4.5, Opus 4.6, GPT-5.1, GPT-5.2, GPT-5.4), with reasoning and 3-sample verdicts, on the same items. Their P1 and P2 prompt rows are this notebook's `vague_criteria` and `no_criteria`. The values were read off the figure in arXiv 2610.02022v3 and are stored in the notebook and in `results/ai2_figure4.csv`.

Optional, off by default: LFM3-1.7B (October 2026) as a possibly contaminated contrast judge, and the judge with the 5 nearest corpus abstracts as related work (Ai2's retrieval condition).

**Held constant:** the benchmark items and a **leakage filter**. Many submissions were on arXiv before the deadline, so any corpus paper matching an idea's OpenReview ID or title, or with cosine similarity of 0.95 or more, is dropped from that idea's neighbours and logged. A paper dropped as a near-copy by bge-small is also dropped for BGE-M3, so both models are scored against the same prior work.

**Two deviations from Ai2's judge setup:** the verdict is read from the probabilities of `0` and `1` after a prefilled, empty `<thinking>` block, instead of sampled reasoning; and each pair runs once in each presentation order, with the two probabilities averaged.

## Metrics

- Pairwise soft accuracy (ties count half) and strict accuracy, and pointwise macro-F1: Ai2's metrics. Embedding scores get a threshold chosen by 5-fold cross-validation. The comparison with Ai2's judges uses strict accuracy, as their Figure 4 does.
- **Uncertainty:** a 95% bootstrap confidence interval and an exact binomial p-value against chance for every pairwise strict accuracy, and exact McNemar tests between conditions on the same pairs (`results/paired_tests.csv`). With six embedding conditions per split, the Bonferroni threshold is p < 0.008.
- **Prompt spread:** how far each judge's pairwise accuracy moves across the three prompts, for this notebook's judge and Ai2's.
- AUROC, the judge's order-flip rate, and seconds per 100 decisions.
- **Quality check:** the Spearman correlation between each score and the mean reviewer rating *within* each pool, to see whether a score tracks quality rather than novelty.

## How to run

1. Open the notebook in Colab with the badge above and pick a **T4 GPU** runtime.
2. No tokens are needed; all data and models are public.
3. Leave `USE_DRIVE = True` to cache the corpus, its embeddings and the judge outputs under `MyDrive/ML/2026-11-novelty-embedding-baseline/`, so a rerun after a disconnect picks up where it stopped. Set it to `False` to cache under `outputs/` in the session instead.
4. Optionally set `SMOKE = True` first (30k-paper corpus, 20 pairs per split) to check everything runs.
5. **Run all.** A first run takes about 1.5–2 hours: about 10 minutes to embed the corpus with bge-small, an estimated 45–90 minutes with BGE-M3, and 15–20 for the judge (about 3,200 forward passes). Everything is cached, so a rerun after a disconnect skips finished work, and a rerun against a complete Drive cache from an earlier full run takes a few minutes.
6. The last cell downloads `results/` and `figures/`. Commit `figures/`, `results/summary.csv`, `results/paired_tests.csv`, `results/comparison_with_ai2.csv`, `results/prompt_spread.csv`, `results/ai2_figure4.csv`, `results/quality_confound.csv` and `results/qualitative_examples.csv`. `results/idea_scores.csv` and `results/leakage_dropped.csv` contain benchmark and corpus titles (ODC-BY and CC BY-SA 4.0); commit them with those licences noted.

## Results

From the full run on 1 November 2026. Pairwise strict accuracy on 154 pairs per split, with 95% bootstrap confidence intervals for the embedding scores. Chance is 0.50.

| Condition | Human-Only | Human+Generated |
|---|---|---|
| Yin, bge-small | 0.51 (0.44–0.59) | 0.36 (0.29–0.44) |
| kNN density, bge-small | 0.49 (0.41–0.57) | 0.40 (0.32–0.47) |
| RND, bge-small | 0.57 (0.49–0.65) | 0.47 (0.40–0.55) |
| **Yin, BGE-M3** | **0.60 (0.53–0.68)** | 0.54 (0.46–0.62) |
| kNN density, BGE-M3 | 0.58 (0.51–0.66) | 0.53 (0.45–0.61) |
| RND, BGE-M3 | 0.55 (0.47–0.63) | 0.71 (0.64–0.79) |
| Qwen2.5-3B judge, three prompts | 0.55–0.58 | 0.06–0.14 |
| Yin (bge-small) + judge | 0.61 | 0.94 |
| Ai2's six frontier judges, reference prompt | 0.74–0.84 | 0.40–0.86 |

- **Human-Only (the main split):** with bge-small, distance is no better than chance. With BGE-M3, Yin reaches 0.60 (binomial p = 0.012) and beats bge-small Yin on the same pairs (28 pairs vs 14, McNemar p = 0.044). That puts it level with the 3B judge on pairs, but none of the embedding scores survives a Bonferroni correction for six conditions (p < 0.008), so treat it as suggestive. Pointwise, the judge separates the pools better (AUROC 0.64–0.68 against 0.57 for BGE-M3 Yin). Everything here is 14–24 points behind Ai2's frontier judges.
- **The two embedding models disagree a lot:** their Yin scores correlate at Spearman 0.64, and they pick the same nearest prior paper for only 19% of ideas.
- **Human+Generated:** BGE-M3 RND's 0.71 is probably not novelty. Every pair sets a human idea against an LLM-written one, RND ranks the generated idea lower in 73% of pairs, and the same metric reaches only 0.55 when both ideas are human. The 3B judge prefers the generated idea in nearly every pair (as Ai2's Sonnet 4.5 does more often than not), and the combined condition's 0.94 comes from the regression inverting the judge, not from the two scores agreeing.
- **Quality check:** within each pool, no embedding score correlates with reviewer rating beyond |ρ| = 0.14, and none significantly.
- **Leakage filter:** 31 matching preprints dropped by OpenReview ID or title, plus 17 bge-small and 7 further BGE-M3 near-copies (cosine ≥ 0.95).

The confidence intervals, paired tests and the Yin (BGE-M3) + judge condition were added to the notebook after this run; rerunning against the Drive cache adds them to `results/` in a few minutes. `results/idea_scores.csv`, `results/leakage_dropped.csv` and `results/qualitative_examples.csv` contain benchmark titles (ODC-BY, Ai2) and corpus titles (CC BY-SA 4.0, Papers with Code archive).

A full write-up will go on the [portfolio site](https://cjosh88.github.io/ai-portfolio/), and this section will then link to it.
