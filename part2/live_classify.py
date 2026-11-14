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


def start_class_query(spark, msgs, model_path, checkpoint_dir):
    raise NotImplementedError("Part 2 owner: implement start_class_query (see docstring).")
