# Amazon Reviews — Sentiment Classification

This repository contains a **single-notebook** NLP project on **sentiment classification** for Amazon product reviews.  
The goal is to compare progressively stronger text representations and models—starting from classical baselines (e.g., **TF-IDF + SVM**) up to contextual encoders with a fine-tuned **DistilBERT**—and to complement performance results with **explainability** techniques (**SHAP** and **gradient-based attribution**).

The task is a **3-class** classification problem: **positive**, **neutral**, **negative**.

---

## What’s inside (all in one notebook)

Everything is implemented and reproducible from the notebook:

- data loading and cleaning
- class balancing (resampling)
- feature extraction (TF-IDF, Word2Vec, GloVe, FastText, Sentence-BERT)
- training/evaluation across multiple classifiers
- transformer fine-tuning (DistilBERT)
- explainability analyses (SHAP + gradient-based attribution)
- tables/plots and the final comparison

> **Main file:** `Amazon_Reviews_Sentiment.ipynb`  
(If your notebook has a different name, update the paths/commands below accordingly.)

---

## Results snapshot

A representative comparison (Accuracy / Macro-F1 / Weighted-F1) shows the expected trend:
- strong classical baseline: **TF-IDF (1–3 grams) + SVM** ≈ **0.83 Accuracy** (Macro-F1 ≈ 0.62)
- best overall model: **DistilBERT (fine-tuned)** ≈ **0.86 Accuracy**, **0.72 Macro-F1**, **0.87 Weighted-F1**

Performance highlights an important practical challenge: the **neutral** class is consistently the hardest to identify, and explainability is used to understand why models confuse ambiguous reviews.

---

## Quickstart

### 1) Create the environment
You can run the notebook locally with Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
