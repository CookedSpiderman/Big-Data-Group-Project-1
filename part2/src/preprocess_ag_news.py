import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("TextPreprocessing").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(args.input)

tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern=r"\W+",
    toLowercase=True
)

tokenized_df = tokenizer.transform(df)

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

cleaned_df = remover.transform(tokenized_df)

print("\nTEXT PREPROCESSING RESULTS:")

sample_rows = cleaned_df.limit(3).collect()

for row in sample_rows:
    print("\nOriginal:", row["text"][:160])
    print("Tokens:", row["words"][:20])
    print("After stop-word removal:", row["filtered_words"][:20])

spark.stop()
