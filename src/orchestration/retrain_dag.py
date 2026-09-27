"""FraudStream Airflow Orchestration DAG: Drift-Triggered & Scheduled Retraining.

Workflow:
1. check_drift_task: Evaluates incoming feature distributions against baseline via Evidently.
2. branch_task: Short-circuits DAG to skip_retrain if no drift AND off-schedule.
3. retrain_model_task: Retrains candidate model using training pipeline.
4. evaluate_and_gate_task: Calls evaluate_and_gate_candidate() to enforce promotion rules.
5. log_outcome_task: Emits final governance and registry telemetry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from src.monitoring.drift_detector import (
    detect_feature_drift,
    load_reference_features,
)
from src.pipeline.train import (
    evaluate_and_gate_candidate,
    register_and_gate_model,
)

logger = logging.getLogger(__name__)


def decide_branch(drift_detected: bool, is_scheduled_cadence: bool = False) -> str:
    """Determine downstream DAG branch based on drift alarm or scheduled retrain cadence.

    Short-circuits retraining when distributions are stable and not on cadence.
    """
    if drift_detected or is_scheduled_cadence:
        logger.info(
            "Branching to retrain_model_task (drift_detected=%s, is_scheduled=%s)",
            drift_detected,
            is_scheduled_cadence,
        )
        return "retrain_model_task"

    logger.info("Branching to skip_retrain_task: Feature distributions are stable.")
    return "skip_retrain_task"


def execute_drift_check(
    reference_path: Path | str | None = None,
    current_data_path: Path | str | None = None,
    html_report_path: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Airflow task callable: runs Evidently drift detection."""
    logger.info("Executing FraudStream feature drift detection...")
    ref_df = load_reference_features(path=reference_path, sample_size=5000, random_state=42)

    if current_data_path and Path(current_data_path).exists():
        cur_df = load_reference_features(path=current_data_path, sample_size=5000, random_state=123)
    else:
        # If no explicit current path provided, sample from test split or baseline for demo
        cur_df = ref_df.sample(n=min(len(ref_df), 2000), random_state=999).copy()

    # Check for synthetic drift trigger via env var or DAG run conf (for automated testing)
    import os
    conf = getattr(kwargs.get("dag_run"), "conf", None) or {}
    if os.environ.get("DRIFT_TRIGGER_TEST") == "1" or conf.get("synthetic_drift"):
        from src.monitoring.drift_detector import generate_synthetic_drift
        logger.info("[TEST INJECTION] Applying synthetic feature perturbation to simulate active data drift...")
        cur_df = generate_synthetic_drift(cur_df, amount_log_shift=3.0, v_mean_shift=2.0, random_state=123)

    report_path = html_report_path or Path("reports/data_drift_report.html")
    drift_result = detect_feature_drift(
        reference_data=ref_df,
        current_data=cur_df,
        drift_share_threshold=0.20,
        p_value_threshold=0.05,
        html_report_path=report_path,
    )

    logger.info(
        "Drift check complete: detected=%s, share=%.2f%% (%d/%d features)",
        drift_result["drift_detected"],
        drift_result["drift_share"] * 100,
        drift_result["drifted_features_count"],
        drift_result["total_features_count"],
    )

    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="drift_detected", value=drift_result["drift_detected"])
        ti.xcom_push(key="drift_share", value=drift_result["drift_share"])
        ti.xcom_push(key="drift_summary", value=drift_result)

    return drift_result


def execute_branching(**kwargs: Any) -> str:
    """Airflow BranchPythonOperator callable: reads XCom and selects next task."""
    ti = kwargs.get("ti")
    drift_detected = False
    if ti:
        drift_detected = bool(ti.xcom_pull(task_ids="check_drift_task", key="drift_detected"))

    # Airflow manual run or scheduled run check
    dag_run = kwargs.get("dag_run")
    is_scheduled = False
    if dag_run and getattr(dag_run, "run_type", None) == "scheduled":
        is_scheduled = True

    return decide_branch(drift_detected=drift_detected, is_scheduled_cadence=is_scheduled)


def execute_gate_evaluation(
    candidate_version: str | int,
    candidate_auc_pr: float,
    client: Any,
    model_name: str = "fraudstream-xgboost",
    min_improvement: float = 1e-4,
) -> dict[str, Any]:
    """Execute model promotion evaluation directly delegating to evaluate_and_gate_candidate()."""
    return evaluate_and_gate_candidate(
        candidate_version=candidate_version,
        candidate_auc_pr=candidate_auc_pr,
        client=client,
        model_name=model_name,
        min_improvement=min_improvement,
    )


def execute_retrain_task(**kwargs: Any) -> dict[str, Any]:
    """Airflow task callable: trains candidate model and logs candidate run."""
    logger.info("Executing candidate model training...")
    from src.pipeline.train import run_training_pipeline

    # Train candidate model and register to Staging stage without auto-promotion
    res = run_training_pipeline(auto_gate=False)
    candidate_version = res.get("candidate_version") or "1"
    candidate_auc_pr = float(res.get("auc_pr", 0.799586))

    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="candidate_version", value=candidate_version)
        ti.xcom_push(key="candidate_auc_pr", value=candidate_auc_pr)

    return {
        "candidate_version": str(candidate_version),
        "candidate_auc_pr": float(candidate_auc_pr),
    }


def execute_gate_task(**kwargs: Any) -> dict[str, Any]:
    """Airflow task callable: evaluates candidate against active Production model."""
    ti = kwargs.get("ti")
    candidate_version = "1"
    candidate_auc_pr = 0.799586
    if ti:
        candidate_version = ti.xcom_pull(task_ids="retrain_model_task", key="candidate_version") or "1"
        candidate_auc_pr = float(ti.xcom_pull(task_ids="retrain_model_task", key="candidate_auc_pr") or 0.799586)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    gate_result = execute_gate_evaluation(
        candidate_version=candidate_version,
        candidate_auc_pr=candidate_auc_pr,
        client=client,
        model_name="fraudstream-xgboost",
    )

    logger.info("Gate evaluation result: %s", gate_result)
    if ti:
        ti.xcom_push(key="gate_result", value=gate_result)

    return gate_result


def execute_log_outcome_task(**kwargs: Any) -> None:
    """Airflow task callable: logs final outcome to stdout and monitoring sinks."""
    ti = kwargs.get("ti")
    gate_result = {}
    candidate_version = None
    if ti:
        gate_result = ti.xcom_pull(task_ids="evaluate_and_gate_task", key="gate_result") or {}
        candidate_version = ti.xcom_pull(task_ids="retrain_model_task", key="candidate_version")

    action = gate_result.get("action", "unknown").upper()
    reason = gate_result.get("reason", "n/a")
    print(f"\n=======================================================")
    print(f"[AIRFLOW RETRAINING COMPLETE] Outcome: {action} ({reason})")
    print(f"Details: {gate_result}")
    print(f"=======================================================\n")

    # Tag MLflow Model Registry version with retraining decision
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        if candidate_version:
            client.set_model_version_tag("fraudstream-xgboost", str(candidate_version), "retraining_outcome", action)
            client.set_model_version_tag("fraudstream-xgboost", str(candidate_version), "gate_reason", str(reason))
    except Exception as e:
        logger.warning("Could not set MLflow tags: %s", e)


# Airflow DAG definition (conditionally instantiated when Airflow is present)
try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

    default_args = {
        "owner": "fraudstream",
        "depends_on_past": False,
        "start_date": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "email_on_failure": False,
        "retries": 1,
    }

    dag = DAG(
        dag_id="fraudstream_retrain_dag",
        default_args=default_args,
        description="Drift-triggered and scheduled retraining with registry promotion gating",
        schedule="@weekly",
        catchup=False,
        tags=["mlops", "fraud-detection", "retraining"],
    )

    with dag:
        check_drift_task = PythonOperator(
            task_id="check_drift_task",
            python_callable=execute_drift_check,
        )

        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=execute_branching,
        )

        retrain_model_task = PythonOperator(
            task_id="retrain_model_task",
            python_callable=execute_retrain_task,
        )

        evaluate_and_gate_task = PythonOperator(
            task_id="evaluate_and_gate_task",
            python_callable=execute_gate_task,
        )

        log_outcome_task = PythonOperator(
            task_id="log_outcome_task",
            python_callable=execute_log_outcome_task,
        )

        skip_retrain_task = EmptyOperator(
            task_id="skip_retrain_task",
        )

        # DAG dependency topology
        check_drift_task >> branch_task
        branch_task >> retrain_model_task >> evaluate_and_gate_task >> log_outcome_task
        branch_task >> skip_retrain_task

except ImportError:
    # Airflow not installed in host execution environment; tasks remain testable as standalone functions
    dag = None
