# CS4371 Group Project 1 — Streaming NER + Text Classification

Team <TeamNumber> — <Name 1 (NetID)>, <Name 2 (NetID)>

- **Part 1:** Bluesky posts → Kafka `topic1` → Spark Structured Streaming (spaCy named entities, running counts) → Kafka `topic2` → Logstash → Elasticsearch → Kibana top-10 bar chart.
- **Part 2:** Spark MLlib pipeline trained on AG News, applied to the streamed posts in batch and live.

No path is hard-coded: every input/output location is a command-line argument. The data source (Bluesky Jetstream) needs no API key. If a keyed source is ever used, the key is read from an environment variable (see `.env.example`) and never written in code.

## Requirements

- Linux, macOS, or Windows with WSL2 (Ubuntu). Spark checkpoints do not work on native Windows without extra Hadoop binaries.
- Docker with Docker Compose
- Java 17 (Spark runs on the JVM)
- Python 3.10–3.12

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Part 1 — Streaming named-entity counts

### 1. Start the infrastructure

```bash
docker compose up -d kafka elasticsearch kibana
# Kafka topics
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic topic1 --partitions 1 --replication-factor 1
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic topic2 --partitions 1 --replication-factor 1
# Elasticsearch field types (entity = exact keyword, count = number)
curl -X PUT localhost:9200/_index_template/entity-counts -H 'Content-Type: application/json' \
  -d '{"index_patterns":["entity-counts*"],"template":{"mappings":{"properties":{"entity":{"type":"keyword"},"count":{"type":"long"}}}}}'
# Logstash reads topic2 and writes one document per entity
docker compose up -d logstash
```

Use `docker compose stop` to pause. `docker compose down` deletes the containers and the topics in them.

### 2. Start the producer (terminal 1)

```bash
python part1/producer.py --bootstrap-servers localhost:9092 --topic topic1 \
    --backup-file <path>/messages.jsonl --sample-rate 0.1
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--bootstrap-servers` | `localhost:9092` | Kafka address |
| `--topic` | `topic1` | Topic the posts are written to |
| `--backup-file` | (required) | JSONL file every sent post is also appended to (Part 2 can read it) |
| `--sample-rate` | `0.1` | Fraction of English posts kept; Jetstream is high volume |
| `--lang` | `en` | Language filter (`""` keeps all) |
| `--max-messages` | `0` | Stop after N posts (0 = run until Ctrl+C) |

Each Kafka message is JSON: `{"id", "text", "created_at", "source"}`.

### 3. Start the Spark streaming app (terminal 2)

```bash
python part1/stream_app.py --bootstrap-servers localhost:9092 \
    --input-topic topic1 --output-topic topic2 \
    --checkpoint-dir <path>/checkpoints/run1 --starting-offsets latest --console-top
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--checkpoint-dir` | (required) | Folder for Spark state; use a new folder for a fresh run |
| `--trigger` | `30 seconds` | Micro-batch interval |
| `--starting-offsets` | `latest` | `latest` = only new posts, `earliest` = everything in topic1 |
| `--max-offsets-per-trigger` | `2000` | Upper bound on posts per micro-batch |
| `--console-top` | off | Also print the top 10 entities to the console each trigger |
| `--model-path` | none | (Part 2) saved PipelineModel; starts the live classification query too |

The Kafka connector JAR matching your PySpark version is downloaded automatically on first run.

Each `topic2` message is `{"entity": "...", "count": N}`, where `count` is the entity's **running total**, not an increment.

### 4. Kibana chart

Open http://localhost:5601.

1. Stack Management → Data Views → create `entity-counts`, timestamp field: *I don't want to use the time filter*.
2. Visualize Library → Lens → Bar horizontal: vertical axis = Top 10 values of `entity` ranked by Maximum of `count`; horizontal axis = Maximum of `count`.
3. Save to a dashboard and set auto-refresh to 10 seconds.

### Starting a fresh run

```bash
curl -X DELETE localhost:9200/entity-counts    # clear old counts (template is kept)
# then start stream_app.py with a NEW --checkpoint-dir
```

## Part 2 — Text classification

_To be completed by the Part 2 owner: training, batch classification, hand-label check, and how to run the live query via `--model-path`._
