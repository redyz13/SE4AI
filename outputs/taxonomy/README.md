# SE4AI Unified Bias Taxonomy

## Overview

This section contains a unified bias taxonomy created for the SE4AI project. The taxonomy is designed to serve as the reference structure for generating pairs of original prompts and counterfactual prompts.

The current taxonomy was built by analyzing three bias-related datasets:

- **CrowS-Pairs**
- **BBQ**
- **HolisticBias**

Each dataset contributed different types of information. CrowS-Pairs was useful for identifying common bias axes and stereotype-related terms. BBQ was useful for understanding bias categories in question answering and decision-oriented settings. HolisticBias provided a broad and systematic lexicon of identity descriptors, sensitive attributes, and demographic terms.

The elements extracted from these datasets were first organized into dataset-specific JSON files. These JSON files were then compared, normalized, and merged into a single definitive taxonomy.

## Goal of the Taxonomy

The goal of the taxonomy is to organize sensitive terms by:

```text
task type → subtask → bias axis → terms
```

This structure makes it possible to identify which bias axes are relevant for each task and subtask, and which terms can be used to generate controlled prompt variations.

For example, the bias axis `nationality` is always named consistently as `nationality` across all tasks and subtasks. The same normalization principle is applied to all other axes, such as `gender`, `race_ethnicity`, `religion`, `age`, `disability`, `socioeconomic_status`, `language_background`, `physical_appearance`, and `sexual_orientation`.

## Source Datasets

### CrowS-Pairs

CrowS-Pairs was used to extract bias categories and terms related to stereotypes and social groups. It contributed useful terms for axes such as:

- `gender`
- `race_ethnicity`
- `nationality`
- `religion`
- `age`
- `disability`
- `socioeconomic_status`
- `physical_appearance`
- `sexual_orientation`

It also provided proxy terms, such as names associated with specific demographic groups.

### BBQ

BBQ was used mainly to support the decision answering side of the taxonomy. It contributed categories, group identifiers, and terms that are useful for ambiguous and disambiguated decision contexts.

It was particularly useful for axes such as:

- `age`
- `disability`
- `gender`
- `nationality`
- `race_ethnicity`
- `religion`
- `socioeconomic_status`

### HolisticBias

HolisticBias was used as the broadest lexical source. It provided a systematic list of descriptors grouped by demographic and identity-related axes.

It contributed terms for axes such as:

- `gender`
- `race_ethnicity`
- `nationality`
- `religion`
- `age`
- `disability`
- `socioeconomic_status`
- `physical_appearance`
- `sexual_orientation`
- `language_background`-related proxies

## Taxonomy Design Choices

### 1. Unified Bias Axis Names

Different datasets use different labels for similar concepts. During the merge process, these labels were normalized into a single set of canonical bias axes.

Examples:

| Dataset labels | Canonical axis |
|---|---|
| `race-color`, `race_ethnicity` | `race_ethnicity` |
| `Gender_identity`, `gender_and_sex`, `gender` | `gender` |
| `Disability_status`, `ability`, `disability` | `disability` |
| `socioeconomic`, `socioeconomic_class` | `socioeconomic_status` |
| `physical-appearance`, `body_type` | `physical_appearance` |
| `sexual-orientation`, `sexual_orientation` | `sexual_orientation` |
| `Nationality`, `nationality` | `nationality` |

This ensures that the same bias axis is always represented with the same name across the whole taxonomy.

### 2. No Buckets in the Final Light Taxonomy

Earlier versions grouped terms into buckets, for example:

```text
nationality → italian → Italian
```

This was removed in the updated version because it made the taxonomy unnecessarily complex for prompt generation. The final light taxonomy directly stores terms under each bias axis:

```text
nationality → ["Italian", "American", "Chinese", "Indian", ...]
```

This is simpler and better suited for automatically rotating terms when creating counterfactual prompts.


### 3. Task and Subtask Organization

The taxonomy is organized around three main task types:

- `classification`
- `recommendation`
- `decision_answering`

Each task type contains a balanced set of subtasks.

#### Classification subtasks

- `hiring_screening`
- `student_evaluation`
- `ticket_priority_classification`
- `eligibility_classification`

#### Recommendation subtasks

- `career_path_recommendation`
- `course_recommendation`
- `role_assignment_recommendation`
- `accessibility_accommodation_recommendation`

#### Decision answering subtasks

- `candidate_comparison`
- `scholarship_allocation`
- `ticket_escalation_decision`
- `public_service_priority_decision`

Each subtask includes only the bias axes that are relevant for that setting.

## Files

### `final_taxonomy_light.json`

This is the simplified taxonomy intended for prompt generation.

Structure:

```json
{
  "classification": {
    "subtasks": {
      "hiring_screening": {
        "gender": ["man", "woman", "male", "female"],
        "nationality": ["Italian", "American", "Chinese", "Indian"]
      }
    }
  }
}
```

This file is the most practical version for generating prompt and counterfactual prompt pairs.

### `final_taxonomy_extended.json`

This is the richer version of the taxonomy. It keeps additional information such as metadata, subtask motivations, sources, and term-level details.

It is useful for documentation, inspection, and future refinement of the taxonomy.

## Intended Use

This unified taxonomy is the final reference layer produced from the analysis of CrowS-Pairs, BBQ, and HolisticBias.

It will serve as the base for generating original prompts and counterfactual prompts. In that process, terms from a given bias axis can be rotated while keeping the rest of the prompt unchanged.

For example:

```text
Original prompt: nationality = Italian
Counterfactual prompt: nationality = Pakistani
```

The same principle can be applied to other axes such as `gender`, `race_ethnicity`, `religion`, `age`, `disability`, `socioeconomic_status`, `language_background`, `physical_appearance`, and `sexual_orientation`.

## Summary

The process followed was:

1. Analyze CrowS-Pairs, BBQ, and HolisticBias.
2. Extract relevant bias axes and sensitive terms from each dataset.
3. Create dataset-specific JSON lexicons.
4. Normalize inconsistent axis names across datasets.
5. Select task-specific and subtask-specific bias axes.
6. Merge all sources into a single definitive taxonomy.
7. Simplify the final structure by removing output formats and buckets.

The resulting taxonomy is now ready to be used as the foundation for prompt vs counterfactual prompt generation.
