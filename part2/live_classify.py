"""
Part 2.3c: live topic classification, run as a SECOND query inside part1/stream_app.py.

stream_app.py calls this when it is started with --model-path:

    start_class_query(spark, msgs, args.model_path, <checkpoint_dir>/classes)

Contract (please keep this signature so the two parts plug together):
    spark           active SparkSession (already has the Kafka connector)
    msgs            STREAMING DataFrame from topic1, already parsed, columns:
                    id, text, created_at, source  (rows with null text removed)
    model_path      folder of the PipelineModel saved by part2/train_model.py
    checkpoint_dir  folder this query must use as its checkpointLocation
                    (separate from the entity query's; two queries cannot share one)
Returns the started StreamingQuery. Do NOT call awaitTermination() here:
stream_app.py waits on all queries with spark.streams.awaitAnyTermination().

Suggested shape:
    model = PipelineModel.load(model_path)
    preds = model.transform(msgs)                      # adds prediction, probability
    labeled = preds.withColumn("label_name", <map 0..3 -> World/Sports/Business/Sci/Tech>)
    counts  = labeled.groupBy("label_name").count()
    return (counts.writeStream.outputMode("complete").format("console")
            .option("checkpointLocation", checkpoint_dir)
            .trigger(processingTime="30 seconds").start())
"""


"""Part 2.3(c): live topic classification alongside the entity-count stream.

The Part 1 application supplies a SparkSession and an already-parsed streaming
DataFrame from Kafka topic1. The returned query runs independently alongside
the Part 1 entity query, using its own checkpoint directory.
"""

from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, trim, when


def start_class_query(spark, msgs, model_path, checkpoint_dir):
    """Start cumulative per-topic predictions for incoming Kafka messages."""
    # Load the same complete fitted preprocessing + classification pipeline.
    model = PipelineModel.load(model_path)

    # Part 1 already removes null texts; also filter out whitespace-only posts.
    clean_msgs = msgs.filter(
        col("text").isNotNull() & (trim(col("text")) != "")
    )

    # Transform is inference: it uses learned weights without training again.
    predictions = model.transform(clean_msgs)

    # The model predicts label IDs 0.0–3.0; make them readable.
    labeled = predictions.withColumn(
        "label_name",
        when(col("prediction") == 0, "World")
        .when(col("prediction") == 1, "Sports")
        .when(col("prediction") == 2, "Business")
        .when(col("prediction") == 3, "Sci/Tech")
        .otherwise("Unknown"),
    )

    # A stateful aggregation: counts keep accumulating across micro-batches.
    counts = labeled.groupBy("label_name").count()

    # Complete mode prints current totals for all classes on every trigger.
    # Use a checkpoint distinct from the Part 1 entity-count query.
    query = (
        counts.writeStream
        .queryName("live_topic_counts")
        .outputMode("complete")
        .format("console")
        .option("checkpointLocation", checkpoint_dir)
        .option("truncate", "false")
        .trigger(processingTime="30 seconds")
        .start()
    )

    # Let stream_app.py manage the lifetime of both streaming queries.
    return query
