# scholar-agent

An evidence-grounded assistant for scientific literature. Given a research question, it retrieves relevant abstracts, extracts key scientific entities, writes a short cited answer, and checks each claim in that answer against the source it cites.

CS6120 Natural Language Processing, final project.

> **Status:** In development. The design is proposed and no experimental results have been measured yet. This README will be updated as components are trained and evaluated.

## Overview

The core contribution is three separately trained transformer components, each owned by one team member and each built around its own methodological question. A single lightweight orchestrator combines them with a pretrained, unmodified text generator into one research workflow.

| Component          | What it does                                               | Research question                                                  | Owner                 |
| ------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------ | --------------------- |
| Retrieval          | Finds abstracts relevant to a question                     | Does training on hard negatives improve ranking?                   | Deepanshu Bhagria     |
| Entity extraction  | Tags methods, tasks, metrics, materials and other entities | Does class-weighted loss improve recognition of rare entity types? | Ashish Gajjela        |
| Claim verification | Judges whether evidence supports or contradicts a claim    | Does neighboring context improve evidence classification?          | Shreyas Suresh Manoti |
| Orchestration      | Connects the components into a cited-answer pipeline       | Do the trained components improve overall system quality?          | Shared                |

## How it works

1. The user asks a research question.
2. The retrieval model ranks abstracts from a local corpus.
3. The entity extraction model tags scientific entities in the retrieved abstracts.
4. Each abstract, its document ID and its entities are packaged as evidence.
5. A pretrained generator writes a concise answer, citing only the supplied document IDs.
6. The answer is split into individual factual claims, each linked to its cited evidence.
7. The verification model labels each claim as supported or contradicted. Claims without suitable evidence are flagged as unverified.
8. The interface shows the answer, references, extracted entities and verification flags.

**Note on verification:** The verifier is a two-class model (support or contradict). It does not learn to recognize missing or irrelevant evidence, so insufficient evidence is flagged conservatively rather than reported as a confident verdict.

## Repository structure

```
data/           Download scripts, splits and validation
retrieval/      Retrieval training, evaluation and inference
extraction/     Entity tagging, training, evaluation and inference
verification/   Evidence pairing, training and inference
agent/          Orchestration, citation validation, prompt templates
tests/          Unit, contract and integration tests
experiments/    Run configurations, results, plots and error samples
docs/           Methodology, dataset notes, final report and slides
```

Datasets and model checkpoints are not stored in git. Use the scripts in `data/` to download data, and see each component's folder for how to obtain or reproduce checkpoints.

## Getting started

```bash
git clone https://github.com/<organization>/scholar-agent.git
cd scholar-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

| Dataset      | Used by            | Download               |
| ------------ | ------------------ | ---------------------- |
| SciERC       | Entity extraction  | `./data/get_scierc.sh` |
| BEIR SciFact | Retrieval          | To be added            |
| SciFact      | Claim verification | To be added            |

## Data contracts

All components exchange data in agreed formats so the pipeline stays traceable from answer back to source.

| Object              | Fields                                                   |
| ------------------- | -------------------------------------------------------- |
| Document            | `doc_id`, `title`, `abstract`, `source_url`              |
| Retrieval result    | `doc_id`, `score`, `text`                                |
| Entity span         | `doc_id`, `entity_type`, `text`, `start`, `end`          |
| Verification result | `claim_id`, `doc_id`, `label`, `score`, `evidence_text`  |
| Answer citation     | `claim_id`, `doc_id`, `evidence_span` or `evidence_text` |

Ground rules:

- Every component preserves the original document IDs.
- Entity offsets refer to the original source text, not a tokenized or normalized copy.
- Missing data produces an explicit empty result or error, never a fabricated citation.

## Contributing

- Each member works in their own component folder on their own branch (for example, `extraction`).
- `main` is protected. Changes go in through pull requests reviewed by a teammate.
- Shared code (data contracts, tests, agent) is changed only after the team agrees.
- Fix random seeds and record configurations for every experiment in `experiments/`.

## Scope

**Required for the final demo:** three trained components with baseline comparisons, a local abstract corpus, a single orchestrator producing cited answers, entity extraction and verification flags, a simple CLI or Streamlit interface, and a reproducible report with ablations.

**Stretch goals:** full-paper PDF ingestion, live arXiv or web search, multi-agent planning, long-term conversational memory, an advanced UI, and larger models or additional datasets.

## Team

| Member                | Role               |
| --------------------- | ------------------ |
| Deepanshu Bhagria     | Retrieval          |
| Ashish Gajjela        | Entity extraction  |
| Shreyas Suresh Manoti | Claim verification |
| Ashish Gajjela        | Integration lead   |

## AI use disclosure

This project follows the CS6120 AI-use disclosure policy. Any AI assistance with code, writing or analysis is documented in `docs/`.

## References

- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BERT-mini: https://huggingface.co/prajjwal1/bert-mini
- BEIR SciFact: https://huggingface.co/datasets/BeIR/scifact
- SciERC: http://nlp.cs.washington.edu/sciIE/ (Luan et al., 2018)
- SciFact: https://github.com/allenai/scifact
- Sentence Transformers: https://www.sbert.net/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
