"""Kafka scoring consumer: reads transactions, scores with XGBoost, publishes results."""

import json
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xgboost as xgb
from confluent_kafka import Consumer, Producer, KafkaError

from src.features.feature_logic import compute_features, FEATURE_COLUMNS

DEFAULT_INPUT_TOPIC = "transactions-raw"
DEFAULT_OUTPUT_TOPIC = "transactions-scored"
DEFAULT_SCORING_GROUP = "fraudstream-scoring-group"
DEFAULT_BOOTSTRAP = "localhost:9092"


class ScoringStats:
    """Thread-safe scoring statistics accumulator."""

    def __init__(self, recent_limit: int = 100):
        self._lock = threading.Lock()
        self.total_scored = 0
        self.total_flagged = 0
        self.latencies: list[float] = []
        self.recent: deque[dict] = deque(maxlen=recent_limit)
        self.start_time = time.time()

    def record(self, scored_payload: dict, latency_ms: float, is_fraud: bool):
        with self._lock:
            self.total_scored += 1
            if is_fraud:
                self.total_flagged += 1
            self.latencies.append(latency_ms)
            self.recent.append(scored_payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lats = np.array(self.latencies) if self.latencies else np.array([0.0])
            return {
                "total_scored": self.total_scored,
                "total_flagged": self.total_flagged,
                "flag_rate": self.total_flagged / max(self.total_scored, 1),
                "latency_ms": {
                    "mean": float(np.mean(lats)),
                    "p50": float(np.median(lats)),
                    "p95": float(np.percentile(lats, 95)),
                    "p99": float(np.percentile(lats, 99)),
                    "min": float(np.min(lats)),
                    "max": float(np.max(lats)),
                },
                "uptime_seconds": time.time() - self.start_time,
            }

    def get_recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return list(self.recent)[-n:]


def score_record(
    record: dict[str, Any],
    model: xgb.XGBClassifier,
    threshold: float,
) -> tuple[dict[str, Any], float]:
    """Score a single transaction record. Returns (scored_payload, latency_ms)."""
    t0 = time.perf_counter()

    features_df = compute_features(record)
    proba = float(model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1][0])
    is_fraud = proba >= threshold

    latency_ms = (time.perf_counter() - t0) * 1000

    scored = {
        **{k: v for k, v in record.items() if k != "Class"},
        "fraud_probability": round(proba, 6),
        "is_fraud": is_fraud,
        "threshold_used": threshold,
        "scoring_latency_ms": round(latency_ms, 3),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    return scored, latency_ms


def run_scoring_consumer(
    model: xgb.XGBClassifier,
    threshold: float,
    stats: ScoringStats,
    input_topic: str = DEFAULT_INPUT_TOPIC,
    output_topic: str = DEFAULT_OUTPUT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    group_id: str = DEFAULT_SCORING_GROUP,
    max_messages: int | None = None,
    idle_timeout: float | None = None,
    stop_event: threading.Event | None = None,
):
    """Consume from input topic, score each record, publish to output topic."""
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    consumer.subscribe([input_topic])
    print(f"[SCORER] Subscribed to '{input_topic}' (group='{group_id}', threshold={threshold})")

    _stop = stop_event or threading.Event()
    last_msg_time = time.time()
    scored_count = 0

    try:
        while not _stop.is_set():
            if max_messages is not None and scored_count >= max_messages:
                break

            msg = consumer.poll(1.0)

            if msg is None:
                if idle_timeout and (time.time() - last_msg_time) > idle_timeout:
                    print(f"[SCORER] Idle timeout ({idle_timeout}s). Stopping.")
                    break
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"[SCORER] Kafka error: {msg.error()}")
                continue

            last_msg_time = time.time()
            record = json.loads(msg.value().decode("utf-8"))

            try:
                scored, latency_ms = score_record(record, model, threshold)
            except Exception as e:
                print(f"[SCORER] Scoring error: {e}")
                continue

            producer.produce(
                output_topic,
                json.dumps(scored).encode("utf-8"),
            )
            producer.poll(0)

            is_fraud = scored["is_fraud"]
            stats.record(scored, latency_ms, is_fraud)
            scored_count += 1

            if is_fraud:
                print(
                    f"[SCORER] ** FRAUD FLAGGED: p={scored['fraud_probability']:.4f} "
                    f"Amount=${record.get('Amount', '?')} Time={record.get('Time', '?')} "
                    f"latency={latency_ms:.1f}ms"
                )

            if scored_count % 500 == 0:
                snap = stats.snapshot()
                print(
                    f"[SCORER] Scored {scored_count:,} | flagged {snap['total_flagged']} "
                    f"| p50={snap['latency_ms']['p50']:.1f}ms p95={snap['latency_ms']['p95']:.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n[SCORER] Interrupted.")
    finally:
        producer.flush()
        consumer.close()
        snap = stats.snapshot()
        print(
            f"[SCORER] Stopped. Scored {snap['total_scored']:,} records, "
            f"flagged {snap['total_flagged']} ({snap['flag_rate']:.2%})"
        )
