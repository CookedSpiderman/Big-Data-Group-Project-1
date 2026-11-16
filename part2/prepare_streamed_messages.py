import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("PrepareStreamedMessages").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the JSONL file
raw_df = spark.read.json(args.input)

print("\nORIGINAL DATASET:")
raw_df.printSchema()
print("Total messages:", raw_df.count())

# Select the relevant columns and clean whitespace
valid_df = raw_df.select(
    "id",
    trim(col("text")).alias("text"),
    "created_at",
    "source"
).filter(
    col("text").isNotNull() & (col("text") != "")
)

# Remove posts with identical text
cleaned_df = valid_df.dropDuplicates(["text"])

valid_count = valid_df.count()
cleaned_count = cleaned_df.count()

print("\nCLEANING RESULTS:")
print("Nonempty messages:", valid_count)
print("Duplicates removed:", valid_count - cleaned_count)
print("Final message count:", cleaned_count)

print("\nSAMPLE CLEANED MESSAGES:")
cleaned_df.select("text").show(3, truncate=90)

# Save the cleaned data as Parquet
cleaned_df.write.mode("overwrite").parquet(args.output)
print("\nCleaned dataset saved:", args.output)

spark.stop()
