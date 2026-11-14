"""
Part 1 producer: Bluesky Jetstream -> Kafka topic1 (+ a JSONL backup file).

Each message sent to Kafka is JSON:
    {"id": "...", "text": "...", "created_at": "...", "source": "bluesky"}

Bluesky Jetstream is public and needs no API key. If you ever switch to a keyed source
(NewsAPI, Finnhub), read the key with os.environ["NEWSAPI_KEY"] -- never hard-code it.

Example:
    python part1/producer.py --bootstrap-servers localhost:9092 --topic topic1 \
        --backup-file data/messages.jsonl --sample-rate 0.1
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

import websockets
from confluent_kafka import Producer

DEFAULT_JETSTREAM = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)


def parse_args():
    p = argparse.ArgumentParser(description="Stream Bluesky posts into Kafka.")
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    p.add_argument("--topic", default="topic1")
    p.add_argument("--backup-file", required=True,
                   help="JSONL file every sent message is appended to (Part 2 can read this).")
    p.add_argument("--sample-rate", type=float, default=0.1,
                   help="Fraction of matching posts to keep (0-1). Jetstream is high volume.")
    p.add_argument("--lang", default="en",
                   help="Keep only posts tagged with this language. Empty string = keep all.")
    p.add_argument("--max-messages", type=int, default=0,
                   help="Stop after this many messages (0 = run until Ctrl+C). Handy for testing.")
    p.add_argument("--jetstream-url", default=DEFAULT_JETSTREAM)
    return p.parse_args()


def to_message(raw, lang):
    """Turn one Jetstream event into our topic1 schema, or None if we skip it."""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if ev.get("kind") != "commit":
        return None
    commit = ev.get("commit") or {}
    if commit.get("operation") != "create" or commit.get("collection") != "app.bsky.feed.post":
        return None
    record = commit.get("record") or {}
    text = (record.get("text") or "").strip()
    if not text:
        return None
    if lang and lang not in (record.get("langs") or []):
        return None
    return {
        "id": f"at://{ev.get('did')}/app.bsky.feed.post/{commit.get('rkey')}",
        "text": text,
        "created_at": record.get("createdAt"),
        "source": "bluesky",
    }


async def stream(args, producer, backup):
    sent = 0
    started = time.time()
    backoff = 1
    while True:
        try:
            async with websockets.connect(args.jetstream_url, ping_interval=20,
                                          max_size=2 ** 22) as ws:
                print(f"Connected to Jetstream. Writing to topic '{args.topic}'.", flush=True)
                backoff = 1
                async for raw in ws:
                    msg = to_message(raw, args.lang)
                    if msg is None or random.random() > args.sample_rate:
                        continue
                    value = json.dumps(msg, ensure_ascii=False)
                    producer.produce(args.topic, key=msg["id"].encode(), value=value.encode())
                    producer.poll(0)              # serve delivery callbacks, keep queue moving
                    backup.write(value + "\n")
                    sent += 1
                    if sent % 100 == 0:
                        rate = sent / (time.time() - started)
                        print(f"{sent} messages sent ({rate:.1f}/sec)", flush=True)
                    if args.max_messages and sent >= args.max_messages:
                        return sent
        except (websockets.ConnectionClosed, OSError) as e:
            # Jetstream connections drop now and then; reconnect with backoff.
            print(f"Connection lost ({e!r}); reconnecting in {backoff}s", file=sys.stderr, flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.backup_file))
    os.makedirs(out_dir, exist_ok=True)

    producer = Producer({"bootstrap.servers": args.bootstrap_servers})
    # buffering=1 -> line-buffered, so the file is usable even if the script is killed.
    with open(args.backup_file, "a", encoding="utf-8", buffering=1) as backup:
        try:
            sent = asyncio.run(stream(args, producer, backup))
            print(f"Done: {sent} messages.")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            producer.flush(10)


if __name__ == "__main__":
    main()
