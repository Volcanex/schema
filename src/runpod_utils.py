"""
RunPod pod and serverless management utilities.
Handles provisioning, monitoring, and teardown of RunPod resources.
"""

import json
import logging
import os
import time
from typing import Optional

import runpod

logger = logging.getLogger(__name__)


def init_runpod(api_key: Optional[str] = None) -> None:
    """Initialise RunPod SDK with API key from arg or env."""
    key = api_key or os.getenv("RUNPOD_API_KEY")
    if not key:
        raise ValueError("RUNPOD_API_KEY not set")
    runpod.api_key = key


# ---------------------------------------------------------------------------
# Pod management (for fine-tuning)
# ---------------------------------------------------------------------------

def list_pods() -> list[dict]:
    """List all pods and their current status."""
    return runpod.get_pods()


def get_pod_status(pod_id: str) -> dict:
    """Get current status of a specific pod."""
    pods = runpod.get_pods()
    for pod in pods:
        if pod["id"] == pod_id:
            return pod
    return {}


def start_training_pod(
    name: str = "schema-model-training",
    gpu_type_id: str = "NVIDIA A100 80GB PCIe",
    container_disk_gb: int = 100,
    volume_in_gb: int = 50,
    image_name: str = "runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
    ports: str = "8888/http",
    volume_mount_path: str = "/workspace",
) -> dict:
    """
    Spin up a RunPod pod for fine-tuning.

    Returns the pod dict. Note: pod takes 1-3 minutes to be RUNNING.
    SSH into it or use the Jupyter Lab URL from the dashboard.
    """
    pod = runpod.create_pod(
        name=name,
        image_name=image_name,
        gpu_type_id=gpu_type_id,
        cloud_type="SECURE",
        container_disk_in_gb=container_disk_gb,
        volume_in_gb=volume_in_gb,
        ports=ports,
        volume_mount_path=volume_mount_path,
        env={
            "JUPYTER_TOKEN": "schema-model",  # Change this
        },
    )
    logger.info(f"Pod started: {pod['id']} — waiting for RUNNING state...")
    return pod


def wait_for_pod(pod_id: str, timeout: int = 300, poll_interval: int = 10) -> bool:
    """Wait until a pod reaches RUNNING state. Returns True if successful."""
    start = time.time()
    while time.time() - start < timeout:
        status = get_pod_status(pod_id)
        desired_status = status.get("desiredStatus", "")
        runtime = status.get("runtime")

        if desired_status == "RUNNING" and runtime:
            logger.info(f"Pod {pod_id} is RUNNING")
            return True

        logger.info(f"Pod {pod_id} status: {desired_status} — waiting...")
        time.sleep(poll_interval)

    logger.error(f"Pod {pod_id} did not start within {timeout}s")
    return False


def stop_pod(pod_id: str) -> None:
    """Stop (not delete) a RunPod pod to pause billing."""
    runpod.stop_pod(pod_id)
    logger.info(f"Pod {pod_id} stopped")


def terminate_pod(pod_id: str) -> None:
    """Permanently delete a RunPod pod."""
    runpod.terminate_pod(pod_id)
    logger.info(f"Pod {pod_id} terminated")


def estimate_training_cost(
    hours: float,
    gpu_type: str = "A100 80GB",
    n_pods: int = 1,
) -> dict:
    """Rough cost estimate for a training run."""
    gpu_rates = {
        "A100 80GB": 1.39,
        "A100 SXM": 1.99,
        "L40S": 0.89,
        "RTX 4090": 0.44,
    }
    rate = gpu_rates.get(gpu_type, 1.39)
    total = rate * hours * n_pods
    return {
        "gpu_type": gpu_type,
        "hours": hours,
        "n_pods": n_pods,
        "rate_per_hour_usd": rate,
        "estimated_cost_usd": round(total, 2),
    }


# ---------------------------------------------------------------------------
# Serverless endpoint management
# ---------------------------------------------------------------------------

def list_endpoints() -> list[dict]:
    """List all serverless endpoints."""
    return runpod.get_endpoints()


def get_endpoint_status(endpoint_id: str) -> dict:
    """Get health and worker status for a serverless endpoint."""
    endpoint = runpod.Endpoint(endpoint_id)
    return endpoint.health()


def submit_serverless_job(
    endpoint_id: str,
    payload: dict,
    wait: bool = True,
    timeout: int = 120,
) -> Optional[dict]:
    """
    Submit a job to a RunPod serverless endpoint.

    Args:
        endpoint_id: The serverless endpoint ID.
        payload: Input dict to send (will be wrapped in {"input": payload}).
        wait: If True, poll until complete and return output.
        timeout: Max seconds to wait.

    Returns:
        Output dict from the handler, or None on failure.
    """
    endpoint = runpod.Endpoint(endpoint_id)

    try:
        run_request = endpoint.run({"input": payload})

        if not wait:
            return {"job_id": run_request.job_id}

        output = run_request.output(timeout=timeout)
        return output
    except Exception as exc:
        logger.error(f"Serverless job failed: {exc}")
        return None


def deploy_serverless(
    name: str = "schema-model-inference",
    docker_image: str = "",  # Your Docker image with the handler
    gpu_type_id: str = "NVIDIA L40S",
    workers_min: int = 0,
    workers_max: int = 4,
    idle_timeout: int = 60,
    container_disk_gb: int = 50,
) -> dict:
    """
    Deploy a serverless endpoint. Requires a Docker image with the RunPod handler.
    See scripts/deploy_serverless.py for the handler implementation.
    """
    if not docker_image:
        raise ValueError(
            "docker_image is required. Build and push your image first:\n"
            "  docker build -t your-registry/schema-model:latest .\n"
            "  docker push your-registry/schema-model:latest"
        )

    endpoint = runpod.create_endpoint(
        name=name,
        image_name=docker_image,
        gpu_type_ids=[gpu_type_id],
        workers_min=workers_min,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
        container_disk_in_gb=container_disk_gb,
    )
    logger.info(f"Serverless endpoint created: {endpoint['id']}")
    return endpoint


def estimate_serverless_cost(
    n_pages: int,
    avg_seconds_per_page: float = 1.44,  # ~2500 pages/hr on L40S
    gpu_type: str = "L40S",
) -> dict:
    """Estimate cost for batch serverless inference."""
    gpu_rates_per_second = {
        "L40S": 0.89 / 3600,
        "A100 80GB": 1.39 / 3600,
        "RTX 4090": 0.44 / 3600,
    }
    rate = gpu_rates_per_second.get(gpu_type, 0.89 / 3600)
    total_seconds = n_pages * avg_seconds_per_page
    total_cost = total_seconds * rate
    hours = total_seconds / 3600

    return {
        "n_pages": n_pages,
        "gpu_type": gpu_type,
        "estimated_hours": round(hours, 1),
        "estimated_cost_usd": round(total_cost, 2),
        "cost_per_page_usd": round(total_cost / n_pages, 5),
    }
