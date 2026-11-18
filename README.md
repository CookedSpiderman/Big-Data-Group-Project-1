# CS4371 Group Project 1 — Streaming NER + Text Classification

Team 55 — Dev Nair (DXN210046), Denilson Montalvo (DAM170000)

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

### 2.1 Data and preprocessing

**Training dataset:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), `train-00000-of-00001.parquet`. It contains **120,000** labeled news articles: **30,000** for each class.

| Label | Class |
| ---: | --- |
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Download the training file to the path used in your command; for example, in Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet" -OutFile ".\data\ag_news_train.parquet"
```

Our original saved Bluesky JSONL file contained **4,050** posts. Spark removed **29** repeated texts (and no blank texts), leaving **4,021 unique nonempty posts** for batch classification.

### 2.2 Training the five-stage MLlib Pipeline

`part2/src/train_classifier.py` builds a pipeline with these stages:

1. `RegexTokenizer` — lowercase text and split it into tokens.
2. `StopWordsRemover` — remove common words that are usually less useful for identifying topics.
3. `HashingTF` — create a sparse, 4,096-position term-frequency vector.
4. `IDF` — learn how common each hashed feature is across the training articles and produce TF-IDF features.
5. Multinomial `LogisticRegression` — estimate probabilities for the four topic labels.

The code uses `maxIter=20` and `regParam=0.1`. The entire fitted `PipelineModel` is saved, which ensures new posts pass through the **same tokenizer, stop-word handling, hash mapping, IDF weights and trained classifier**.

To **retrain** and save a model in the repository layout:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace cs4371-spark-full:3.5.8 /opt/spark/bin/spark-submit --master "local[2]" --driver-memory 3g part2/src/train_classifier.py --input data/ag_news_train.parquet --model-output part2/ag_news_classifier
```

> **Caution:** This script saves with overwrite mode. If the checked-in model is already present and works, training again is optional and will replace the model directory at the output path.

### 2.3 Clean and classify the saved posts (batch)

If your producer backup is saved at `data/bluesky_live.jsonl`, run:

```powershell
# Remove blank text and duplicate text; write Parquet
docker run --rm -v "${PWD}:/workspace" -w /workspace cs4371-spark-full:3.5.8 /opt/spark/bin/spark-submit --master "local[2]" part2/src/prepare_streamed_messages.py --input data/bluesky_live.jsonl --output data/cleaned_posts

# Apply the saved PipelineModel and save predictions
docker run --rm -v "${PWD}:/workspace" -w /workspace cs4371-spark-full:3.5.8 /opt/spark/bin/spark-submit --master "local[2]" --driver-memory 3g part2/src/classify_posts.py --input data/cleaned_posts --model part2/ag_news_classifier --output part2/outputs/batch_predictions
```

For the original **4,021-message** batch, the observed predicted counts were:

| Predicted class | Posts |
| --- | ---: |
| World | 471 |
| Sports | 1,115 |
| Business | 589 |
| Sci/Tech | 1,846 |
| **Total** | **4,021** |

The batch output includes the message text, predicted class, the largest predicted probability (`confidence`), and four separate class-probability columns (`p_world`, `p_sports`, `p_business`, `p_scitech`). Spark writes CSV into an **output directory** such as `part2/outputs/batch_predictions/`, typically containing a `part-*.csv` file. The script also displays **five sample predictions per class** with probabilities.

### 2.4 Random sample, human labels and accuracy

Create a 50-message sample for manual annotation:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace cs4371-spark-full:3.5.8 /opt/spark/bin/spark-submit --master "local[2]" part2/src/create_evaluation_sample.py --input part2/outputs/batch_predictions --output part2/outputs/hand_labels_50.csv
```

Open the resulting CSV and fill `human_label` with **World**, **Sports**, **Business**, **Sci/Tech** or **None of these**. Label based on the content without looking at the model's guess first. **Do not rerun the sampling script over a completed hand-label file**; keep the annotated original for submission.

Calculate and export the evaluation:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace cs4371-spark-full:3.5.8 python3 part2/src/evaluate_50.py --input part2/outputs/hand_labels_50.csv --summary part2/outputs/evaluation_summary.txt --errors part2/outputs/evaluation_errors.csv
```

Results from the hand-labeled sample:

- Randomly selected messages: **50**
- **39** labeled *None of these*; **11** fit one of the four news categories
- **4** correct and **7** incorrect predictions among those 11
- **Accuracy on applicable messages = 4 / 11 = 36.36%**

The *None of these* cases are reported separately because the classifier was trained only on four classes. Its accuracy estimate is based on just 11 applicable examples and should not be treated as a precise overall accuracy estimate. Many informal Bluesky conversations are unlike the AG News training articles (**domain shift**), so the model sometimes assigns an unrelated news class to personal posts or comments. The errors CSV contains specific misclassifications for discussion in the report.

### 2.5 Real-time prediction (second Structured Streaming query)

`part2/live_classify.py` defines:

```python
def start_class_query(spark, msgs, model_path, checkpoint_dir):
    # Returns the started StreamingQuery.
    ...
```

`part1/stream_app.py` imports this function when invoked with `--model-path`. It receives the **already-parsed streaming DataFrame from `topic1`**, loads the saved PipelineModel, predicts one of four classes for each new message, and displays **cumulative counts by class** with `writeStream` in `complete` output mode. The class stream uses its **own checkpoint**, separate from the entity stream. It runs alongside Part 1, not as a replacement for Part 1.

Use the **Terminal 1** Spark command in the Docker quick start above to run both queries. The class-count console displays fields `label_name` and `count`. Our local four-message integration test showed one message classified in each of the four classes; a later real-data test showed growing counts while the producer was running. These checks demonstrate operation; the full live run is the source for final demonstration screenshots.

## Output files and final submission

Expected deliverables include:

- Python source code for **Part 1 and Part 2**, plus Docker/Compose and Logstash configuration.
- The original or reproducibly collected message file (JSONL) and instructions to obtain the AG News training data.
- The fitted PipelineModel directory or instructions/commands to regenerate it.
- Batch prediction CSV containing **every message's predicted class and four probabilities**.
- `hand_labels_50.csv`, with 50 human labels beside the model predictions.
- `evaluation_summary.txt` and `evaluation_errors.csv` with evaluation statistics and example mistakes.
- Kibana bar-chart screenshots from **multiple actual time intervals**, showing the top 10 named entities and their cumulative counts.
- Project report PDF describing the source, methods, results and limitations; include the team contribution table, an accessible link to a narrated **8-minute-maximum** video where each member explains their work, and the required AI-use appendix.
- If AI assistants were used, complete conversation transcripts as PDFs in an `ai_transcripts/` folder.

Package the required contents into one ZIP named `<TeamNumber>-GP1.zip` for the final submission. **Do not put API keys, tokens or secrets in the repository or ZIP.** Fill in the team placeholders at the top of this README before submitting.

## Stopping, restarting and troubleshooting

- To stop the **producer** and **Spark** streams, use `Ctrl+C` in their respective terminals. Close neither window during active recording or collection.
- Run `docker compose stop` to stop the infrastructure **without removing containers**. Restart with `docker compose up -d`.
- `docker compose down` removes containers and the Compose network. Without persistent Kafka storage, it can also remove stored Kafka topics/messages; use `stop` when you want to preserve your local test state.
- If Spark says `Failed to find data source: kafka`, use `spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8` as shown in the tested command.
- If Spark cannot find the model, verify the entire `part2/ag_news_classifier/metadata` directory exists and that the provided path is relative to `/workspace` (the mounted repository).
- If Kafka works from Windows but not from Spark's Docker container, check the container is on the Compose network and uses `kafka:29092`, **not** `localhost:9092`.
- If Kibana displays old test results, check the saved Elasticsearch documents, that Logstash is running, and the Lens configuration (**Top values of `entity.keyword`**, **Maximum of `count`**). New Spark checkpoints alone do **not** erase old Elasticsearch documents.
- A failed command inside Docker does not delete the Python scripts, data or saved model in the mounted repository. Keep `data/`, `part2/outputs/`, and `part2/ag_news_classifier/` backed up.

## References

- [Apache Spark Structured Streaming](https://spark.apache.org/docs/3.5.8/structured-streaming-programming-guide.html)
- [Spark Structured Streaming + Kafka integration](https://spark.apache.org/docs/3.5.8/structured-streaming-kafka-integration.html)
- [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Bluesky Jetstream](https://docs.bsky.app/blog/jetstream)
