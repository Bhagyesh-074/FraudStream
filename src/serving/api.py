"""FastAPI observability layer for FraudStream scoring service."""

import argparse
import threading

import uvicorn
from fastapi import FastAPI

from src.serving.model_loader import (
    load_model,
    load_threshold_config,
    get_active_model_source,
)
from src.serving.scoring_consumer import ScoringStats, run_scoring_consumer

app = FastAPI(title="FraudStream Scoring Service", version="1.0.0")

_stats: ScoringStats | None = None
_threshold: float | None = None
_model_loaded: bool = False


def get_stats() -> ScoringStats:
    global _stats
    if _stats is None:
        _stats = ScoringStats()
    return _stats


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model_loaded,
        "threshold": _threshold,
        "model_source": get_active_model_source(),
    }


@app.get("/stats")
def stats():
    return get_stats().snapshot()


@app.get("/recent")
def recent(n: int = 20):
    return get_stats().get_recent(n)


def start_service(
    host: str = "0.0.0.0",
    port: int = 8000,
    bootstrap_servers: str = "localhost:9092",
    idle_timeout: float | None = None,
    max_messages: int | None = None,
):
    """Start scoring consumer in background thread + FastAPI server."""
    global _stats, _threshold, _model_loaded

    model = load_model()
    config = load_threshold_config()
    _threshold = config["optimal_threshold"]
    _model_loaded = True
    _stats = ScoringStats()

    print(f"[SERVICE] Model loaded. Threshold: {_threshold}")

    stop_event = threading.Event()
    consumer_thread = threading.Thread(
        target=run_scoring_consumer,
        kwargs={
            "model": model,
            "threshold": _threshold,
            "stats": _stats,
            "bootstrap_servers": bootstrap_servers,
            "idle_timeout": idle_timeout,
            "max_messages": max_messages,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    consumer_thread.start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        stop_event.set()
        consumer_thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudStream scoring service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--idle-timeout", type=float, default=None)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()

    start_service(
        host=args.host,
        port=args.port,
        bootstrap_servers=args.bootstrap,
        idle_timeout=args.idle_timeout,
        max_messages=args.max_messages,
    )
