"""Unit and integration tests for Kafka streaming producer and raw consumer."""

import json
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.eda.download_data import DEFAULT_CSV_PATH
from src.streaming.producer import serialize_record, replay_stream
from src.streaming.raw_consumer import (
    get_partition_dir,
    flush_batch_to_parquet,
    consume_to_raw_zone,
)

EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def is_kafka_running(host: str = "localhost", port: int = 9092) -> bool:
    """Check whether Kafka broker is reachable on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


class TestSerialization:
    """Test serialization of transaction records for Kafka streaming."""

    def test_serialize_record_valid_json(self):
        """Record must serialize to valid JSON string."""
        record = {col: 0.0 for col in EXPECTED_COLUMNS}
        record["Class"] = 0
        serialized = serialize_record(record)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
        assert len(parsed) == 31

    def test_serialize_record_preserves_time_and_amount(self):
        """Serialization must preserve historical Time and Amount exactly."""
        record = {"Time": 402.5, "Amount": 122.21, "Class": 1}
        parsed = json.loads(serialize_record(record))
        assert parsed["Time"] == 402.5
        assert parsed["Amount"] == 122.21
        assert parsed["Class"] == 1

    def test_serialize_record_keys_match_csv_schema(self):
        """Serialized record keys must match all 31 expected CSV columns."""
        record = {col: 1.0 for col in EXPECTED_COLUMNS}
        parsed = json.loads(serialize_record(record))
        assert set(parsed.keys()) == set(EXPECTED_COLUMNS)


class TestRawZoneParquet:
    """Test raw Parquet file generation, partitioning, and immutability."""

    def test_partition_directory_format(self, tmp_path):
        """Partition directory must follow date=YYYY-MM-DD/hour=HH format."""
        dt = datetime(2026, 9, 21, 14, 30, 0, tzinfo=timezone.utc)
        part_dir = get_partition_dir(tmp_path, dt)
        assert "date=2026-09-21" in str(part_dir)
        assert "hour=14" in str(part_dir)
        assert part_dir.is_dir()

    def test_flush_empty_buffer_returns_none(self, tmp_path):
        """Flushing empty buffer should return None and create no file."""
        assert flush_batch_to_parquet([], tmp_path) is None

    def test_flush_batch_creates_parquet_file(self, tmp_path):
        """Flushing buffer creates valid Parquet file with expected rows."""
        buffer = [{"Time": float(i), "Amount": float(i * 10), "Class": 0} for i in range(10)]
        pq_path = flush_batch_to_parquet(buffer, tmp_path)
        assert pq_path is not None and pq_path.is_file()
        df = pd.read_parquet(pq_path)
        assert len(df) == 10
        assert list(df["Time"]) == [float(i) for i in range(10)]

    def test_parquet_immutability(self, tmp_path):
        """Raw values written to Parquet must be byte/value faithful with no transformation."""
        buffer = [
            {"Time": 0.0, "V1": -1.359807, "Amount": 149.62, "Class": 0},
            {"Time": 1.0, "V1": 1.191857, "Amount": 2.69, "Class": 0},
        ]
        pq_path = flush_batch_to_parquet(buffer, tmp_path)
        df_read = pd.read_parquet(pq_path)
        for i, row in enumerate(buffer):
            assert df_read.iloc[i]["Time"] == row["Time"]
            assert abs(df_read.iloc[i]["V1"] - row["V1"]) < 1e-6
            assert df_read.iloc[i]["Amount"] == row["Amount"]
            assert df_read.iloc[i]["Class"] == row["Class"]


class TestKafkaStreamingIntegration:
    """Integration test verifying end-to-end Kafka produce and consume."""

    @pytest.fixture(autouse=True)
    def check_kafka(self):
        """Skip integration tests if local Kafka broker is unreachable."""
        if not is_kafka_running():
            pytest.skip("Local Kafka broker (localhost:9092) not reachable. Run 'docker compose up -d'.")
        if not DEFAULT_CSV_PATH.is_file():
            pytest.skip(f"Dataset CSV not found at {DEFAULT_CSV_PATH}")

    def test_end_to_end_subset_stream(self, tmp_path):
        """Run producer & consumer concurrently for 500 records; verify Parquet fidelity."""
        topic = f"test-stream-{int(time.time() * 1000)}"
        group = f"test-group-{int(time.time() * 1000)}"
        n_records = 500

        consumer_thread = threading.Thread(
            target=consume_to_raw_zone,
            kwargs={
                "topic": topic,
                "group_id": group,
                "raw_zone_dir": tmp_path,
                "batch_size": 250,
                "max_messages": n_records,
                "poll_timeout": 1.0,
                "idle_timeout": 15.0,
            },
        )
        consumer_thread.start()
        time.sleep(2.0)

        producer_thread = threading.Thread(
            target=replay_stream,
            kwargs={
                "csv_path": DEFAULT_CSV_PATH,
                "topic": topic,
                "delay_seconds": 0.001,
                "limit": n_records,
                "log_interval": 250,
            },
        )
        producer_thread.start()

        producer_thread.join(timeout=30.0)
        consumer_thread.join(timeout=30.0)

        parquet_files = list(tmp_path.glob("**/*.parquet"))
        assert len(parquet_files) >= 1

        df_recovered = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        assert len(df_recovered) == n_records

        df_source = pd.read_csv(DEFAULT_CSV_PATH, nrows=n_records)
        assert list(df_recovered.columns) == list(df_source.columns)

        # Confirm Time column preserved unmodified
        df_recovered_sorted = df_recovered.sort_values("Time").reset_index(drop=True)
        pd.testing.assert_series_equal(
            df_recovered_sorted["Time"],
            df_source["Time"].astype(float),
            check_names=True,
            check_exact=False,
            rtol=1e-5,
        )
        # Confirm Amount column preserved unmodified
        pd.testing.assert_series_equal(
            df_recovered_sorted["Amount"],
            df_source["Amount"],
            check_names=True,
            check_exact=False,
            rtol=1e-5,
        )
