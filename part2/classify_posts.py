import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, array_max
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("ClassifyBlueskyPosts").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the cleaned Bluesky posts
posts = spark.read.parquet(args.input)

# Load the previously trained model
model = PipelineModel.load(args.model)

# Predict topics and probabilities
predictions = model.transform(posts)

# Convert probability vectors to regular arrays
predictions = predictions.withColumn(
    "probabilities",
    vector_to_array("probability")
)

# Create readable output columns
results = (
    predictions
    .withColumn("predicted_label", col("prediction").cast("int"))
    .withColumn(
        "predicted_class",
        when(col("predicted_label") == 0, "World")
        .when(col("predicted_label") == 1, "Sports")
        .when(col("predicted_label") == 2, "Business")
        .when(col("predicted_label") == 3, "Sci/Tech")
        .otherwise("Unknown")
    )
    .withColumn("confidence", array_max(col("probabilities")))
    .withColumn("p_world", col("probabilities")[0])
    .withColumn("p_sports", col("probabilities")[1])
    .withColumn("p_business", col("probabilities")[2])
    .withColumn("p_scitech", col("probabilities")[3])
    .select(
        "id", "text", "created_at", "source",
        "predicted_label", "predicted_class",
        "confidence",
        "p_world", "p_sports", "p_business", "p_scitech"
    )
)

results = results.cache()

print("\nTOTAL CLASSIFIED POSTS:", results.count())

print("\nCLASS COUNTS:")
results.groupBy("predicted_class").count().orderBy(
    "predicted_class"
).show()

classes = ["World", "Sports", "Business", "Sci/Tech"]

for category in classes:
    print("\nEXAMPLES FOR:", category)
    results.filter(col("predicted_class") == category).select(
        "text", "confidence"
    ).show(5, truncate=90)

# Save all predictions and probabilities to CSV
results.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(args.output)

print("\nPredictions saved:", args.output)

results.unpersist()
spark.stop()
