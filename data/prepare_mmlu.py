"""
Download MMLU from HuggingFace and convert to debate topic format.
Output: data/mmlu_topics.json  — a list of topic strings, one per question.

Usage:
    python data/prepare_mmlu.py                         # all subjects, 100 samples
    python data/prepare_mmlu.py --subjects anatomy law  # specific subjects
    python data/prepare_mmlu.py --n 500                 # more samples
    python data/prepare_mmlu.py --split test            # use test split
"""

import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset

CHOICES = ["A", "B", "C", "D"]

ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def to_debate_topic(row: dict) -> str:
    question = row["question"].strip()
    choices = row["choices"]
    options = "\n".join(f"{CHOICES[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{question}\n{options}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="MMLU subjects to include (default: all)")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--n", type=int, default=100,
                        help="Total number of topics to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: data/mmlu_topics.json)")
    args = parser.parse_args()

    random.seed(args.seed)
    subjects = args.subjects or ALL_SUBJECTS
    output_path = Path(args.output) if args.output else Path(__file__).parent / "mmlu_topics.json"

    print(f"Loading MMLU ({args.split} split) for {len(subjects)} subject(s)...")

    all_topics = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=args.split)
            for row in ds:
                all_topics.append({
                    "topic": to_debate_topic(row),
                    "subject": subject,
                    "answer": CHOICES[row["answer"]],
                })
        except Exception as e:
            print(f"  [skip] {subject}: {e}")

    random.shuffle(all_topics)
    selected = all_topics[: args.n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} topics → {output_path}")
    print(f"Subject coverage: {len({t['subject'] for t in selected})} subjects")


if __name__ == "__main__":
    main()
