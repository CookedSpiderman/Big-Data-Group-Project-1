import argparse
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("AGNewsLoading").getOrCreate()

df = spark.read.parquet(args.input)

print("DATASET STRUCTURE:")
df.printSchema()

print("TOTAL ARTICLES:", df.count())

print("ARTICLES PER CLASS:")
df.groupBy("label").count().orderBy("label").show()

print("SAMPLE ARTICLES:")
df.select("text", "label").show(3, truncate=90)

spark.stop()
