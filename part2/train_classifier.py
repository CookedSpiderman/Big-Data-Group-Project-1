import argparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--model-output", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("AGNewsClassifier").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the labeled AG News dataset
df = spark.read.parquet(args.input).select("text", "label").dropna()

print("Training articles:", df.count())

# Stage 1: Tokenization
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern=r"\W+",
    toLowercase=True
)

# Stage 2: Stop-word removal
remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

# Stage 3: Term frequency
tf = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=4096
)

# Stage 4: Inverse document frequency
idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

# Stage 5: Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    maxIter=20,
    regParam=0.1,
    family="multinomial"
)

# Build the complete pipeline
pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, lr])

print("Training classifier...")
model = pipeline.fit(df)

# Save the fitted pipeline
model.write().overwrite().save(args.model_output)
print("MODEL SAVED:", args.model_output)

# Test predictions on new sentences
samples = spark.createDataFrame([
    ("The president met with world leaders to discuss peace.",),
    ("The basketball team won the championship game.",),
    ("Company shares increased after quarterly earnings.",),
    ("Scientists developed a new artificial intelligence model.",)
], ["text"])

predictions = model.transform(samples).select(
    "text", "prediction", "probability"
).collect()

categories = ["World", "Sports", "Business", "Sci/Tech"]

print("\nSAMPLE PREDICTIONS:")
for row in predictions:
    predicted_class = int(row["prediction"])
    probabilities = [round(float(p), 3) for p in row["probability"]]

    print("\nText:", row["text"])
    print("Predicted category:", categories[predicted_class])
    print("Probabilities:", probabilities)

spark.stop()
