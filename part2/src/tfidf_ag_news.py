import argparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("TFIDF_AGNews").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(args.input)

tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern=r"\W+",
    toLowercase=True
)

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

tf = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=4096
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])

model = pipeline.fit(df)
result = model.transform(df)

row = result.select(
    "text", "filtered_words", "rawFeatures", "features"
).first()

print("\nTF-IDF RESULTS:")
print("Original:", row["text"][:150])
print("Filtered words:", row["filtered_words"][:15])
print("Number of feature positions:", row["features"].size)
print("Nonzero feature count:", len(row["features"].indices))

print("\nFirst 8 feature indices and TF-IDF weights:")
for index, value in zip(row["features"].indices[:8], row["features"].values[:8]):
    print("Feature index:", index, "Weight:", round(float(value), 3))

spark.stop()
