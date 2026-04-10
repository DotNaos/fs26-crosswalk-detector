"""Small local JSON server for server-side crosswalk scan jobs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Event, Lock
from typing import Any
import uuid
from urllib.parse import urlparse

from .scan_backend import (
    CLIP_MODEL_ID,
    DETECTOR_MODEL_ID,
    HybridCrosswalkScanner,
    SceneRequest,
    TileRequest,
    _sigmoid,
    _tile_pixel_bounds,
    crop_tile,
    crosswalk_score_image,
    decide_tile_label,
    detector_overlap_score,
    fetch_scene_image,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_error(message: str, status: HTTPStatus) -> tuple[int, dict[str, Any]]:
    return status.value, {"ok": False, "error": message}


@dataclass
class ScanJob:
    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    total: int = 0
    done: int = 0
    current_tile_id: str | None = None
    suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    cancel_event: Event = field(default_factory=Event, repr=False)
    lock: Lock = field(default_factory=Lock, repr=False)

    def update(self, **changes: Any) -> None:
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)
            self.updated_at = _utc_now()

    def add_suggestion(self, tile_id: str, suggestion: dict[str, Any], done: int) -> None:
        with self.lock:
            self.suggestions[tile_id] = suggestion
            self.done = done
            self.current_tile_id = tile_id
            self.updated_at = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "scene_id": self.scene_id,
                "status": self.status,
                "stage": self.stage,
                "total": self.total,
                "done": self.done,
                "current_tile_id": self.current_tile_id,
                "results": self.suggestions,
                "error": self.error,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


class ScanService:
    def __init__(self) -> None:
        self._scanner: HybridCrosswalkScanner | None = None
        self._scanner_lock = Lock()
        self._jobs: dict[str, ScanJob] = {}
        self._jobs_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._warmup_executor = ThreadPoolExecutor(max_workers=1)
        self._warmup_future: Future[HybridCrosswalkScanner] | None = None

    def scanner(self) -> HybridCrosswalkScanner:
        with self._scanner_lock:
            if self._scanner is None:
                self._scanner = HybridCrosswalkScanner()
            return self._scanner

    def ready(self) -> bool:
        with self._scanner_lock:
            return self._scanner is not None

    def warmup(self) -> None:
        with self._scanner_lock:
            if self._scanner is not None:
                return
            if self._warmup_future and not self._warmup_future.done():
                return
            self._warmup_future = self._warmup_executor.submit(self.scanner)

    def health(self) -> dict[str, Any]:
        scanner = self._scanner
        return {
            "ok": True,
            "ready": scanner is not None,
            "warming": bool(self._warmup_future and not self._warmup_future.done()),
            "model": f"{os.path.basename(DETECTOR_MODEL_ID)} + {os.path.basename(CLIP_MODEL_ID)}",
            "device": scanner.device if scanner else "not-loaded",
            "busy": any(job.status in {"queued", "running"} for job in self._jobs.values()),
        }

    def start(self, request: dict[str, Any]) -> ScanJob:
        scene_payload = request.get("scene") or {}
        tiles_payload = request.get("tiles") or []
        if not scene_payload.get("scene_id"):
            raise ValueError("Missing scene payload.")
        if not isinstance(tiles_payload, list) or not tiles_payload:
            raise ValueError("Missing tiles payload.")
        job = ScanJob(job_id=str(uuid.uuid4()), scene_id=str(scene_payload["scene_id"]), total=len(tiles_payload))
        with self._jobs_lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run_job, job, request)
        return job

    def get(self, job_id: str) -> ScanJob:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        job.cancel_event.set()
        job.update(status="cancelled", stage="cancelled")

    def _run_job(self, job: ScanJob, request: dict[str, Any]) -> None:
        try:
            scene_payload = request["scene"]
            tiles_payload = request["tiles"]
            threshold = float(request.get("threshold", 0.32))
            scene = SceneRequest(
                scene_id=str(scene_payload["scene_id"]),
                latitude=float(scene_payload["latitude"]),
                longitude=float(scene_payload["longitude"]),
                size_m=int(scene_payload["size_m"]),
                image_px=int(scene_payload["image_px"]),
                tile_size_m=int(scene_payload["tile_size_m"]),
            )
            tiles = [
                TileRequest(
                    tile_id=str(tile["tile_id"]),
                    row=int(tile["row"]),
                    col=int(tile["col"]),
                    bbox_mercator=tuple(float(value) for value in tile["bbox_mercator"]),
                    relative_path=str(tile["relative_path"]),
                )
                for tile in tiles_payload
            ]

            job.update(status="running", stage="loading-models")
            scanner = self.scanner()
            if job.cancel_event.is_set():
                job.update(status="cancelled", stage="cancelled")
                return

            job.update(stage="downloading-scene")
            scene_image = fetch_scene_image(scene)
            if job.cancel_event.is_set():
                job.update(status="cancelled", stage="cancelled")
                return

            job.update(stage="detecting-crosswalks")
            detections = scanner.detect_boxes(scene_image)
            if job.cancel_event.is_set():
                job.update(status="cancelled", stage="cancelled")
                return

            job.update(stage="scoring-tiles")
            context_images = [crop_tile(scene_image, scene, tile, padding_tiles=1.0) for tile in tiles]
            center_images = [crop_tile(scene_image, scene, tile) for tile in tiles]
            clip_scores = scanner.score_context_images(context_images)

            for index, tile in enumerate(tiles, start=1):
                if job.cancel_event.is_set():
                    job.update(status="cancelled", stage="cancelled")
                    return
                tile_bounds = _tile_pixel_bounds(scene, tile.bbox_mercator)
                detector_score = detector_overlap_score(detections, tile_bounds)
                heuristic_probability = _sigmoid(crosswalk_score_image(center_images[index - 1]) / 8.0)
                clip_positive, clip_negative = clip_scores[index - 1]
                metrics = decide_tile_label(clip_positive, clip_negative, heuristic_probability, detector_score, threshold)
                job.add_suggestion(
                    tile.tile_id,
                    {
                        "tile_id": tile.tile_id,
                        "label": metrics.label,
                        "score": round(metrics.combined_score, 6),
                        "peak": round(max(metrics.clip_positive, metrics.detector_score), 6),
                        "coverage": round(metrics.heuristic_probability, 6),
                        "prompt": "server-hybrid",
                        "selected": True,
                        "review_source": "python-hybrid-scan",
                    },
                    index,
                )

            job.update(status="completed", stage="completed", current_tile_id=None)
        except Exception as exc:  # pragma: no cover
            job.update(status="failed", stage="failed", error=str(exc))


service = ScanService()


class ScanRequestHandler(BaseHTTPRequestHandler):
    server_version = "CrosswalkScanServer/1.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, service.health())
            return
        if path.startswith("/scan/"):
            job_id = path.removeprefix("/scan/")
            if "/" in job_id:
                self._send_json(*_json_error("Unknown route.", HTTPStatus.NOT_FOUND))
                return
            try:
                self._send_json(HTTPStatus.OK, service.get(job_id).snapshot())
            except KeyError:
                self._send_json(*_json_error(f"Unknown job: {job_id}", HTTPStatus.NOT_FOUND))
            return
        self._send_json(*_json_error("Unknown route.", HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/warmup":
            service.warmup()
            self._send_json(HTTPStatus.OK, service.health())
            return
        if path == "/scan/start":
            try:
                payload = self._read_json()
                job = service.start(payload)
                self._send_json(HTTPStatus.OK, {"job_id": job.job_id, "status": job.status})
            except ValueError as exc:
                self._send_json(*_json_error(str(exc), HTTPStatus.BAD_REQUEST))
            except Exception as exc:  # pragma: no cover
                self._send_json(*_json_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR))
            return
        if path.startswith("/scan/") and path.endswith("/cancel"):
            job_id = path.removeprefix("/scan/").removesuffix("/cancel").strip("/")
            try:
                service.cancel(job_id)
                self._send_json(HTTPStatus.OK, {"ok": True})
            except KeyError:
                self._send_json(*_json_error(f"Unknown job: {job_id}", HTTPStatus.NOT_FOUND))
            return
        self._send_json(*_json_error("Unknown route.", HTTPStatus.NOT_FOUND))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(body.decode("utf-8"))

    def _send_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)


def main() -> None:
    host = os.getenv("CROSSWALK_SCAN_HOST", "127.0.0.1")
    port = int(os.getenv("CROSSWALK_SCAN_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), ScanRequestHandler)
    print(f"Crosswalk scan server on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["main", "service"]


if __name__ == "__main__":
    main()
