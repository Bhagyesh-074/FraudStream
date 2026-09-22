"""Kafka producer replaying transaction records in chronological order."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from confluent_kafka import Producer

DEFAULT_TOPIC = "transactions-raw"
DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "creditcard.csv"


def serialize_record(record: dict[str, Any]) -> str:
    """Serialize transaction record to JSON, preserving raw Time and values."""
    return json.dumps(record)


def create_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP) -> Producer:
    """Create confluent_kafka Producer instance."""
    return Producer({"bootstrap.servers": bootstrap_servers, "client.id": "fraudstream-producer"})


def replay_stream(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    topic: str = DEFAULT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    delay_seconds: float = 0.01,
    limit: int | None = None,
    start_row: int = 0,
    log_interval: int = 500,
    on_delivery: Callable | None = None,
) -> int:
    """Replay CSV transactions to Kafka in chronological order."""
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    producer = create_producer(bootstrap_servers)
    df = pd.read_csv(csv_file)
    if start_row > 0:
        df = df.iloc[start_row:]
    if limit is not None:
        df = df.iloc[:limit]

    records = df.to_dict(orient="records")
    total = len(records)
    print(f"[PRODUCER] Starting replay of {total:,} records to topic '{topic}' (delay={delay_seconds}s)")

    sent_count = 0
    start_time = time.time()

    def _default_delivery_callback(err, msg):
        if err:
            print(f"[PRODUCER] Message delivery failed: {err}")

    callback = on_delivery or _default_delivery_callback

    for i, record in enumerate(records, start=1):
        payload = serialize_record(record)
        producer.produce(topic, payload.encode("utf-8"), callback=callback)
        producer.poll(0)
        sent_count += 1

        if sent_count % log_interval == 0 or sent_count == total:
            elapsed = time.time() - start_time
            rate = sent_count / elapsed if elapsed > 0 else 0
            print(f"[PRODUCER] Sent {sent_count:,}/{total:,} messages ({rate:.1f} msg/s)")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    producer.flush()
    elapsed = time.time() - start_time
    print(f"[PRODUCER] Finished replay: {sent_count:,} messages sent in {elapsed:.2f}s")
    return sent_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay credit card transactions to Kafka")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH), help="Path to creditcard.csv")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Kafka topic name")
    parser.add_argument("--bootstrap", type=str, default=DEFAULT_BOOTSTRAP, help="Bootstrap servers")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between messages in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to replay")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row index (0-based) in CSV")
    parser.add_argument("--log-interval", type=int, default=500, help="Log progress every N messages")
    args = parser.parse_args()

    replay_stream(
        csv_path=args.csv,
        topic=args.topic,
        bootstrap_servers=args.bootstrap,
        delay_seconds=args.delay,
        limit=args.limit,
        start_row=args.start_row,
        log_interval=args.log_interval,
    )
