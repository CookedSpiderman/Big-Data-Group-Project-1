"""
Part 1 Spark Structured Streaming app.

Query 1 (Part 1): topic1 -> named entities (spaCy) -> running count per entity -> topic2
                  Each topic2 message: {"entity": "Apple", "count": 42}
Query 2 (Part 2, optional): pass --model-path and the live topic classifier from
                  part2/live_classify.py is started in this same application.

Example:
    python part1/stream_app.py --bootstrap-servers localhost:9092 \
        --input-topic topic1 --output-topic topic2 \
        --checkpoint-dir checkpoints/run1 --console-top
"""
import argparse
import os
import re
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, struct, to_json, udf, desc
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

# Schema of the JSON our producer writes to topic1.
TOPIC1_SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("text", StringType()),
    StructField("created_at", StringType()),
    StructField("source", StringType()),
])

# spaCy entity labels we count: people, organizations, places, groups, events, products.
KEEP_LABELS = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "PRODUCT"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_POSSESSIVE = re.compile(r"(?:'s|’s)$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Running named-entity counts from Kafka to Kafka.")
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    p.add_argument("--input-topic", default="topic1")
    p.add_argument("--output-topic", default="topic2")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Base folder for checkpoints. Use a NEW folder for a fresh run.")
    p.add_argument("--trigger", default="30 seconds", help="Micro-batch interval.")
    p.add_argument("--starting-offsets", default="latest", choices=["latest", "earliest"],
                   help="latest = only new posts; earliest = also count everything already in topic1.")
    p.add_argument("--max-offsets-per-trigger", type=int, default=2000,
                   help="Cap on messages per micro-batch so spaCy never gets a huge backlog at once.")
    p.add_argument("--console-top", action="store_true",
                   help="Also print the current top 10 entities to the console at each trigger.")
    p.add_argument("--model-path", default=None,
                   help="(Part 2) Saved PipelineModel. If given, also starts the live classification query.")
    return p.parse_args()


# ---------- Named-entity extraction (runs on the Spark workers) ----------

_nlp = None  # loaded lazily, once per Python worker process (loading spaCy per row would be very slow)


def clean_entity(text):
    """Normalize one entity string; return None if it isn't worth counting."""
    t = _URL.sub("", text)
    t = " ".join(t.split())                       # collapse newlines / repeated spaces
    t = t.strip(" \t\"'`.,:;!?()[]{}<>#*-_|/\\“”‘’")
    t = _POSSESSIVE.sub("", t).strip()            # "Apple's" -> "Apple"
    if len(t) < 2 or t.isdigit() or t.startswith("@"):
        return None
    if not any(ch.isalpha() for ch in t):         # drop things like "2024-11" or emoji-only
        return None
    return t


def extract_entities(text):
    global _nlp
    if not text:
        return []
    if _nlp is None:
        import spacy
        # We only need the NER component; skipping the parser/lemmatizer makes it much faster.
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    out = []
    for ent in _nlp(text).ents:
        if ent.label_ in KEEP_LABELS:
            cleaned = clean_entity(ent.text)
            if cleaned:
                out.append(cleaned)
    return out


entities_udf = udf(extract_entities, ArrayType(StringType()))


# ---------- Spark plumbing ----------

def build_spark():
    kafka_pkg = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"
    return (SparkSession.builder
            .appName("GP1-entity-stream")
            .config("spark.jars.packages", kafka_pkg)     # Kafka connector matching your Spark version
            # Default is 200 shuffle partitions; for a small local stream that means 200 tiny
            # state stores per trigger. 4 keeps each micro-batch fast.
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate())


def read_messages(spark, args):
    """topic1 as a streaming DataFrame of parsed messages (id, text, created_at, source)."""
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", args.bootstrap_servers)
           .option("subscribe", args.input_topic)
           .option("startingOffsets", args.starting_offsets)
           .option("maxOffsetsPerTrigger", args.max_offsets_per_trigger)
           .option("failOnDataLoss", "false")
           .load())
    return (raw.selectExpr("CAST(value AS STRING) AS json")      # Kafka values arrive as bytes
            .select(from_json(col("json"), TOPIC1_SCHEMA).alias("m"))
            .select("m.*")
            .where(col("text").isNotNull()))


def entity_counts(msgs):
    return (msgs.select(explode(entities_udf(col("text"))).alias("entity"))
            .groupBy("entity")
            .count())


def main():
    args = parse_args()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    msgs = read_messages(spark, args)
    counts = entity_counts(msgs)

    # Query 1: running totals -> topic2. "update" mode sends only entities whose total
    # changed in this micro-batch; each message carries the NEW total (a replacement, not a delta).
    to_kafka = (counts
                .select(col("entity").alias("key"),
                        to_json(struct("entity", "count")).alias("value"))
                .writeStream
                .queryName("entity_counts_to_kafka")
                .format("kafka")
                .option("kafka.bootstrap.servers", args.bootstrap_servers)
                .option("topic", args.output_topic)
                .option("checkpointLocation", os.path.join(args.checkpoint_dir, "entities"))
                .outputMode("update")
                .trigger(processingTime=args.trigger)
                .start())
    print(f"Started: {args.input_topic} -> entity counts -> {args.output_topic}")

    if args.console_top:
        # Debug view: sorting a streaming aggregation is only allowed in "complete" mode.
        (counts.orderBy(desc("count"))
         .writeStream
         .queryName("entity_top10_console")
         .format("console")
         .option("numRows", 10)
         .option("truncate", "false")
         .option("checkpointLocation", os.path.join(args.checkpoint_dir, "console_top"))
         .outputMode("complete")
         .trigger(processingTime=args.trigger)
         .start())

    if args.model_path:
        # Part 2.3c: second query in the same application (written by the Part 2 owner).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from part2.live_classify import start_class_query
        start_class_query(spark, msgs, args.model_path,
                          os.path.join(args.checkpoint_dir, "classes"))
        print("Started: live topic classification query")

    # Wait on ALL queries, so the app keeps running as long as any of them does.
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
