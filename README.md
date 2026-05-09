# 🐈‍⬛‍️ SE4AI — Counterfactual Fairness Testing for Generative AI 🦇

## 📌 Overview

This repository contains an early-stage project developed as part of the **SE4AI course**.

The project focuses on the analysis of **bias in generative language models**, with particular attention to how sensitive features in the input may influence model outputs.

The main goal is to build a pipeline for **counterfactual fairness testing**: starting from neutral prompts, the system generates controlled variants by modifying only sensitive attributes such as gender, ethnicity, religion, or other relevant features. The model outputs are then compared to evaluate whether these changes affect the model behavior.

The project is currently in a preliminary phase and will evolve as the methodology, implementation, and experiments become more mature.

---

## 🎯 Goals

The project aims to:

* Identify relevant sensitive features from existing literature, datasets, and benchmarks
* Organize these features into a structured taxonomy
* Generate counterfactual prompts by modifying sensitive attributes in a controlled way
* Evaluate how these variations influence generative model outputs
* Study mitigation strategies to reduce observed bias

The focus is on building a repeatable process to detect and analyze biased behavior in generative AI systems.

---

## 🧠 Research Context

Large Language Models are widely used in many application domains, but their outputs may be affected by sensitive attributes present in the input.

Traditional fairness evaluation often relies on static benchmarks, which may be limited in coverage and generalizability. This project adopts a **counterfactual testing** perspective: a model should ideally behave consistently when sensitive attributes that are irrelevant to the task are changed.

The project is inspired by research on:

* Counterfactual fairness
* Bias detection in language models
* Benchmark-based fairness evaluation
* Automatic prompt-based test generation
* Bias mitigation through prompt and input reformulation

---

## 🧪 Methodology

The proposed approach follows a pipeline based on controlled prompt generation and output comparison.

### 1️⃣ Sensitive Feature Analysis

Existing datasets and benchmarks, such as **CrowS-Pairs** and **StereoSet**, are analyzed to identify sensitive features commonly associated with biased model behavior.

These features are organized into categories such as:

* Gender
* Ethnicity
* Religion
* Nationality
* Age
* Other context-dependent sensitive attributes

---

### 2️⃣ Counterfactual Prompt Generation

Starting from neutral base prompts, the system generates counterfactual variants by modifying only the selected sensitive feature.

The generation process may use:

* Rule-based transformations
* Direct attribute replacement
* Controlled support from language models

The objective is to obtain prompt variants that are semantically equivalent and differ only in the sensitive attribute under analysis.

---

### 3️⃣ Model Execution and Output Comparison

The selected generative model is executed on both the original prompts and their counterfactual variants.

The outputs are compared to evaluate whether the model remains stable or changes its behavior in ways that may suggest bias.

The project plans to define an operational measure of **fairness compliance**, inspired by the idea of compliance in prompt-based test generation. This measure is intended to quantify how consistently the model behaves when irrelevant sensitive attributes are changed.

---

### 4️⃣ Bias Mitigation

When potentially biased behavior is detected, mitigation strategies are applied and evaluated.

Possible strategies include:

* Prompt rewriting
* Input reformulation
* More explicit neutralization of task instructions
* Re-execution of the evaluation pipeline after mitigation

---

## 🛠️ Repository Status

This repository is currently under active development.

---

## 👥 Credits

This project is developed as part of a SE4AI course.

The repository represents an ongoing effort to apply software engineering principles to the evaluation and improvement of fairness in generative AI systems.