"""
inspect_data.py

Inspects the processed SciERC splits (train, dev, test) before they are
converted to BIO labels for token classification. For each split it
reports document, sentence and token counts, entity counts per type,
span lengths, overlapping or nested spans, and sanity checks on the
span indices. It ends with a class-imbalance summary computed on the
training split only.

SciERC format (verified on the processed JSON):
    - One document per line, with keys: doc_key, sentences, ner,
      relations, clusters.
    - `sentences` is a list of token lists, one per sentence.
    - `ner` has one list per sentence of [start, end, label] spans.
    - start and end are DOCUMENT-level token indices (they keep counting
      across sentences), and `end` is INCLUSIVE.

Usage:
    python extraction/inspect_data.py --data-dir <folder_with_split_files>
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# Names of the three official SciERC splits (files are <split>.json)
SPLITS = ("train", "dev", "test")

# Number of overlap examples to print per split
MAX_EXAMPLES = 5

# Number of index errors to print per split (all are still counted)
MAX_ERRORS_SHOWN = 10


def load_split(path):
    """
    Load one SciERC split file, where each line is one JSON document.

    Args:
        path (Path): Path to the split file (e.g. train.json).

    Returns:
        list[dict]: One dictionary per document.
    """
    with open(path, encoding="utf-8") as f:
        # Skip blank lines (e.g. a trailing newline at end of file)
        return [json.loads(line) for line in f if line.strip()]


def classify_pair(a, b):
    """
    Decide how two entity spans relate to each other.

    Args:
        a (list): First span as [start, end, label], end inclusive.
        b (list): Second span as [start, end, label], end inclusive.

    Returns:
        str or None: 'nested' if one span lies inside the other
        (including identical spans), 'crossing' if they partially
        overlap, or None if they do not overlap at all.
    """
    (s1, e1, _), (s2, e2, _) = a, b

    # No shared tokens: one span ends before the other starts
    if e1 < s2 or e2 < s1:
        return None

    # One span fully contains the other
    if (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2):
        return "nested"

    # They share some tokens but neither contains the other
    return "crossing"


def inspect_split(docs):
    """
    Collect statistics and sanity checks for one split.

    Args:
        docs (list[dict]): Documents returned by load_split().

    Returns:
        dict: Counts, Counters, overlap totals, index errors and a few
        printable overlap examples for the split.
    """
    stats = {
        "docs": len(docs),
        "sentences": 0,
        "tokens": 0,
        "entities": 0,
        "types": Counter(),           # entity label -> count
        "span_lengths": Counter(),    # span length in tokens -> count
        "nested_pairs": 0,
        "crossing_pairs": 0,
        "identical_span_diff_label": 0,
        "sentences_with_overlap": 0,
        "index_errors": [],
        "examples": [],
    }

    for doc in docs:
        # Flatten the document so document-level indices map to tokens
        tokens = [t for sent in doc["sentences"] for t in sent]

        # Every sentence should have its own (possibly empty) span list
        if len(doc["ner"]) != len(doc["sentences"]):
            stats["index_errors"].append(f"{doc['doc_key']}: ner/sentence count mismatch")

        offset = 0  # document-level index of the current sentence's first token
        for sent, spans in zip(doc["sentences"], doc["ner"]):
            sent_start, sent_end = offset, offset + len(sent) - 1
            stats["sentences"] += 1
            stats["tokens"] += len(sent)

            # Count each entity and check it stays inside its sentence
            for start, end, label in spans:
                stats["entities"] += 1
                stats["types"][label] += 1
                stats["span_lengths"][end - start + 1] += 1  # +1 because end is inclusive
                if not (sent_start <= start <= end <= sent_end):
                    stats["index_errors"].append(
                        f"{doc['doc_key']}: span [{start}, {end}] outside its sentence"
                    )

            # Compare every pair of spans in the sentence for overlaps
            has_overlap = False
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    kind = classify_pair(spans[i], spans[j])
                    if kind is None:
                        continue
                    has_overlap = True
                    stats[f"{kind}_pairs"] += 1

                    # Same boundaries but different labels would be a conflict
                    if spans[i][:2] == spans[j][:2]:
                        stats["identical_span_diff_label"] += 1

                    # Keep a few readable examples for the report
                    if len(stats["examples"]) < MAX_EXAMPLES:
                        a, b = spans[i], spans[j]
                        stats["examples"].append(
                            f"[{kind}] {a[2]}: '{' '.join(tokens[a[0]:a[1] + 1])}'  vs  "
                            f"{b[2]}: '{' '.join(tokens[b[0]:b[1] + 1])}'"
                        )

            stats["sentences_with_overlap"] += has_overlap  # True counts as 1
            offset += len(sent)  # move to the next sentence's first token

    return stats


def report(name, s):
    """
    Print a readable summary of one split's statistics.

    Args:
        name (str): Split name, used as the section heading.
        s (dict): Statistics returned by inspect_split().
    """
    print(f"\n=== {name} ===")
    print(f"Documents: {s['docs']}   Sentences: {s['sentences']}   "
          f"Tokens: {s['tokens']}   Entities: {s['entities']}")

    # Entity types, most frequent first, with their share of all entities
    print("\nEntities per type:")
    for label, n in s["types"].most_common():
        print(f"  {label:<22}{n:>6}  ({100 * n / s['entities']:5.1f}%)")

    # Span lengths: how many entities are single tokens or short phrases
    total = s["entities"]
    lengths = s["span_lengths"]
    short = sum(n for length, n in lengths.items() if length <= 3)
    print(f"\nSpan length: 1 token {100 * lengths[1] / total:.1f}%, "
          f"<=3 tokens {100 * short / total:.1f}%, max {max(lengths)} tokens")

    # Overlaps: these decide the policy for flattening spans into BIO tags
    print(f"\nOverlaps: {s['nested_pairs']} nested pairs, {s['crossing_pairs']} crossing pairs, "
          f"{s['identical_span_diff_label']} identical spans with different labels")
    print(f"Sentences containing any overlap: {s['sentences_with_overlap']} / {s['sentences']}")
    for ex in s["examples"]:
        print(f"  e.g. {ex}")

    # Index checks: any error here means the conversion step would mislabel tokens
    if s["index_errors"]:
        print(f"\nIndex problems ({len(s['index_errors'])}):")
        for err in s["index_errors"][:MAX_ERRORS_SHOWN]:
            print(f"  {err}")
    else:
        print("\nIndex checks passed: every span lies within its sentence.")


def report_imbalance(train_stats):
    """
    Print the ratio between the most and least frequent entity types.

    Uses the training split only, because class weights for the
    weighted-loss experiment must be computed from training data alone
    to avoid leakage from dev or test.

    Args:
        train_stats (dict): Statistics for the train split from inspect_split().
    """
    ranked = train_stats["types"].most_common()
    (most_label, most_n), (least_label, least_n) = ranked[0], ranked[-1]
    print(f"\n=== Train class imbalance ===\n"
          f"Most frequent: {most_label} ({most_n}), least frequent: {least_label} ({least_n}), "
          f"ratio {most_n / least_n:.1f}x")


def main():
    """
    Entry point: parse the data directory argument, inspect and report
    each split, then print the training-set class-imbalance summary.
    """
    parser = argparse.ArgumentParser(description="Inspect the processed SciERC splits.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing train.json, dev.json and test.json")
    args = parser.parse_args()

    # Inspect each split once and keep the results for the imbalance summary
    all_stats = {}
    for split in SPLITS:
        all_stats[split] = inspect_split(load_split(args.data_dir / f"{split}.json"))
        report(split, all_stats[split])

    report_imbalance(all_stats["train"])


if __name__ == "__main__":
    main()