"""Kafka raw consumer writing immutable transaction micro-batches to Parquet."""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from confluent_kafka import Consumer, KafkaError

DEFAULT_TOPIC = "transactions-raw"
DEFAULT_GROUP_ID = "fraudstream-raw-consumer-group"
DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_RAW_ZONE = Path(__file__).resolve().parent.parent.parent / "data" / "raw_zone"


def create_consumer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    group_id: str = DEFAULT_GROUP_ID,
    auto_offset_reset: str = "earliest",
) -> Consumer:
    """Create confluent_kafka Consumer instance."""
    return Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": auto_offset_reset,
        "enable.auto.commit": True,
    })


def get_partition_dir(base_dir: Path, wall_clock_dt: datetime) -> Path:
    """Compute partition directory using ingestion wall-clock date and hour."""
    date_str = wall_clock_dt.strftime("%Y-%m-%d")
    hour_str = wall_clock_dt.strftime("%H")
    partition_dir = base_dir / f"date={date_str}" / f"hour={hour_str}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    return partition_dir


def flush_batch_to_parquet(buffer: list[dict[str, Any]], base_dir: Path) -> Path | None:
    """Flush in-memory record buffer to partitioned Parquet file without transformation."""
    if not buffer:
        return None

    now = datetime.now(timezone.utc)
    target_dir = get_partition_dir(base_dir, now)
    batch_filename = f"batch_{int(now.timestamp() * 1000)}_{uuid.uuid4().hex[:8]}.parquet"
    filepath = target_dir / batch_filename

    table = pa.Table.from_pylist(buffer)
    pq.write_table(table, filepath)
    return filepath


def consume_to_raw_zone(
    topic: str = DEFAULT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    group_id: str = DEFAULT_GROUP_ID,
    raw_zone_dir: str | Path = DEFAULT_RAW_ZONE,
    batch_size: int = 500,
    max_messages: int | None = None,
    poll_timeout: float = 1.0,
    idle_timeout: float | None = None,
) -> int:
    """Consume raw messages from Kafka and write immutable Parquet files."""
    base_dir = Path(raw_zone_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    consumer = create_consumer(bootstrap_servers, group_id)
    consumer.subscribe([topic])
    print(f"[CONSUMER] Subscribed to topic '{topic}' (group='{group_id}', batch_size={batch_size})")

    buffer: list[dict[str, Any]] = []
    total_written = 0
    last_msg_time = time.time()
    start_time = time.time()

    try:
        while True:
            if max_messages is not None and total_written + len(buffer) >= max_messages:
                break

            msg = consumer.poll(poll_timeout)

            if msg is None:
                if idle_timeout and (time.time() - last_msg_time) > idle_timeout:
                    print(f"[CONSUMER] Idle timeout ({idle_timeout}s) reached without new messages. Stopping.")
                    break
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"[CONSUMER] Kafka error: {msg.error()}")
                continue

            last_msg_time = time.time()
            record = json.loads(msg.value().decode("utf-8"))
            buffer.append(record)

            if len(buffer) >= batch_size:
                parquet_path = flush_batch_to_parquet(buffer, base_dir)
                total_written += len(buffer)
                print(f"[CONSUMER] Flushed {len(buffer)} records -> {parquet_path.name} (total: {total_written:,})")
                buffer.clear()

        if buffer:
            parquet_path = flush_batch_to_parquet(buffer, base_dir)
            total_written += len(buffer)
            print(f"[CONSUMER] Final flush {len(buffer)} records -> {parquet_path.name} (total: {total_written:,})")
            buffer.clear()

    except KeyboardInterrupt:
        print("\n[CONSUMER] Interrupted by user, flushing pending records...")
        if buffer:
            flush_batch_to_parquet(buffer, base_dir)
            total_written += len(buffer)
            buffer.clear()
    finally:
        consumer.close()
        elapsed = time.time() - start_time
        print(f"[CONSUMER] Consumer closed. Total records written to raw zone: {total_written:,} in {elapsed:.2f}s")

    return total_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume Kafka transactions to raw Parquet zone")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Kafka topic name")
    parser.add_argument("--bootstrap", type=str, default=DEFAULT_BOOTSTRAP, help="Bootstrap servers")
    parser.add_argument("--group-id", type=str, default=DEFAULT_GROUP_ID, help="Consumer group ID")
    parser.add_argument("--output", type=str, default=str(DEFAULT_RAW_ZONE), help="Output directory for raw zone")
    parser.add_argument("--batch-size", type=int, default=500, help="Micro-batch size for Parquet flush")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after consuming N messages")
    parser.add_argument("--idle-timeout", type=float, default=None, help="Stop if idle for N seconds")
    args = parser.parse_args()

    consume_to_raw_zone(
        topic=args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        raw_zone_dir=args.output,
        batch_size=args.batch_size,
        max_messages=args.max_messages,
        idle_timeout=args.idle_timeout,
    )
