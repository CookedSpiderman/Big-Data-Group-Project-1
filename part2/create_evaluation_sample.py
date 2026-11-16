import argparse
import csv
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("EvaluationSample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the predictions from Step 10
df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .csv(args.input)
)

# Randomly select 50 messages
sample = (
    df.select("id", "text", "predicted_class", "confidence")
    .orderBy(rand(seed=42))
    .limit(50)
    .collect()
)

# Prevent accidentally overwriting hand labels
if os.path.exists(args.output):
    raise FileExistsError("Output already exists: " + args.output)

# Save an editable CSV
with open(args.output, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow([
        "id", "text", "human_label",
        "predicted_class", "confidence"
    ])

    for row in sample:
        writer.writerow([
            row["id"],
            row["text"],
            "",
            row["predicted_class"],
            row["confidence"]
        ])

print("\nEVALUATION SAMPLE CREATED")
print("Messages selected:", len(sample))
print("Saved to:", args.output)

spark.stop()
