import argparse
import csv
from collections import Counter
from pathlib import Path

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
VALID_HUMAN_LABELS = set(CATEGORIES + ["None of these"])

parser = argparse.ArgumentParser(description="Evaluate 50 hand-labeled streamed messages")
parser.add_argument("--input", required=True)
parser.add_argument("--summary", required=True)
parser.add_argument("--errors", required=True)
args = parser.parse_args()

with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    required = {"id", "text", "human_label", "predicted_class", "confidence"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("CSV must have columns: " + ", ".join(sorted(required)))
    rows = list(reader)

if len(rows) != 50:
    raise ValueError(f"Expected 50 labeled messages, found {len(rows)}")
if len({row["id"] for row in rows}) != len(rows):
    raise ValueError("Duplicate post IDs found in the evaluation sample")

for i, row in enumerate(rows, start=2):
    row["human_label"] = row["human_label"].strip()
    row["predicted_class"] = row["predicted_class"].strip()
    if row["human_label"] not in VALID_HUMAN_LABELS:
        raise ValueError(f"Invalid or missing human_label at CSV row {i}: {row['human_label']!r}")
    if row["predicted_class"] not in CATEGORIES:
        raise ValueError(f"Invalid predicted_class at CSV row {i}: {row['predicted_class']!r}")

applicable = [r for r in rows if r["human_label"] != "None of these"]
errors = [r for r in applicable if r["human_label"] != r["predicted_class"]]
correct = len(applicable) - len(errors)
none_count = len(rows) - len(applicable)
accuracy = correct / len(applicable) if applicable else None
human_counts = Counter(r["human_label"] for r in rows)
prediction_counts = Counter(r["predicted_class"] for r in rows)

summary_lines = [
    "CS4371 GP1 - Manual Classification Evaluation",
    f"Total random messages: {len(rows)}",
    f"None of these: {none_count}",
    f"Applicable messages: {len(applicable)}",
    f"Correct predictions: {correct}",
    f"Incorrect predictions: {len(errors)}",
    f"Four-class accuracy: {accuracy:.2%}" if accuracy is not None else "Four-class accuracy: undefined (no applicable messages)",
    "",
    "Human label counts:",
    *[f"  {name}: {human_counts[name]}" for name in CATEGORIES + ["None of these"]],
    "",
    "Model prediction counts in sample:",
    *[f"  {name}: {prediction_counts[name]}" for name in CATEGORIES],
    "",
    "Note: 'None of these' messages are excluded from four-class accuracy.",
    "Accuracy is a small-sample estimate; labels may be subjective.",
]

for output in (args.summary, args.errors):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(args.summary).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
with open(args.errors, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "human_label", "predicted_class", "confidence"])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in errors)

print("\n".join(summary_lines))
print("\nSaved summary:", args.summary)
print("Saved misclassified posts:", args.errors)
