#!/usr/bin/env python3
"""Small local web API that runs the real ashry PyTorch trainers."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from pointclouds import compare, export, export_3dgs, export_3dgs_mesh, target_cloud, write_ply


ASHRY = Path(__file__).resolve().parents[1]
MANIFEST = ASHRY / "benchmark" / "results" / "benchmark_v1.json"
app = Flask(__name__, static_folder=str(Path(__file__).parent))
jobs: dict[str, dict] = {}
lock = threading.Lock()


@app.after_request
def allow_local_file_ui(response):
    """Allow the bundled page to call this local-only server when opened as a file."""
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def manifest_data() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def valid_object(object_name: str) -> bool:
    return any(
        f'{record["category"]}/{record["object_id"]}' == object_name
        for record in manifest_data()["objects"]
    )


def record_for(object_name: str) -> dict:
    for record in manifest_data()["objects"]:
        if f'{record["category"]}/{record["object_id"]}' == object_name:
            return record
    raise ValueError(f"Unknown benchmark object: {object_name}")


def job_output(method: str, object_name: str) -> Path:
    slug = object_name.replace("/", "_")
    return ASHry_outputs(method) / slug


def ASHry_outputs(method: str) -> Path:
    return ASHRY / method / "results" / "app"


def job_output_path(job: dict) -> Path:
    path = Path(job["output"])
    return path if path.is_absolute() else ASHRY / path


def clear_previous_output(output: Path) -> None:
    """Prevent a new run from exposing artifacts produced by an older run."""
    for name in ("checkpoint.pt", "pointcloud.ply", "pointcloud_metrics.json", "gaussian_centers.npy", "app.log"):
        (output / name).unlink(missing_ok=True)
    for name in ("heldout",):
        shutil.rmtree(output / name, ignore_errors=True)


def command_for(method: str, object_name: str, iterations: int, image_size: int, output: Path) -> tuple[list[str], Path]:
    manifest = str(MANIFEST.resolve())
    common = ["--manifest", manifest, "--object", object_name, "--iterations", str(iterations),
              "--image-size", str(image_size)]
    if method == "3dgs":
        return ([sys.executable, "train_3dgs.py", *common, "--save-every", "100",
                 "--output-dir", str(output.parent)], ASHRY / "3dgs")
    if method == "nerf":
        return ([sys.executable, "train.py", *common, "--output-dir", str(output), "--rays-per-step", "1024",
                 "--coarse-samples", "32", "--fine-samples", "32", "--render-chunk", "2048"], ASHRY / "nerf")
    if method == "neus":
        return ([sys.executable, "train.py", *common, "--output-dir", str(output), "--rays-per-step", "1024",
                 "--coarse-samples", "32", "--fine-samples", "32", "--render-chunk", "2048",
                 "--eikonal-points", "512", "--checkpoint-every", "100"], ASHRY / "neus")
    raise ValueError("method must be 3dgs, nerf, or neus")


def read_tail(path: Path, count: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-count:])


def run_job(job_id: str, command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n\n")
        stream.flush()
        try:
            process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except OSError as error:
            with lock:
                jobs[job_id].update(status="failed", error=str(error), finished=time.time())
            return
        with lock:
            jobs[job_id]["pid"] = process.pid
        return_code = process.wait()
        with lock:
            jobs[job_id].update(status="completed" if return_code == 0 else "failed",
                                return_code=return_code, finished=time.time())


@app.get("/api/objects")
def objects():
    data = manifest_data()
    return jsonify({"dataset_dir": data["dataset_dir"], "objects": [
        {"name": f'{r["category"]}/{r["object_id"]}', "split": r["split"],
         "input_views": r["input_views"], "heldout_views": r["heldout_views"]}
        for r in data["objects"]
    ]})


@app.get("/api/ground-truth")
def ground_truth():
    object_name = request.args.get("object", "")
    view = request.args.get("view", "023.png")
    if not valid_object(object_name):
        return jsonify(error="Unknown benchmark object."), 404
    record = record_for(object_name)
    if view not in record["input_views"] + record["heldout_views"]:
        return jsonify(error="Unknown view for this object."), 400
    cloud_path = ASHRY / "app" / "reference" / f"{object_name.replace('/', '_')}.ply"
    if not cloud_path.is_file():
        write_ply(cloud_path, target_cloud(manifest_data(), object_name))
    render_path = Path(manifest_data()["dataset_dir"]) / record["render_dir"] / view
    return jsonify({
        "cloud": str(cloud_path.relative_to(ASHRY)),
        "image": f"/api/ground-truth/image?object={object_name}&view={view}",
        "view": view,
    })


@app.get("/api/ground-truth/image")
def ground_truth_image():
    object_name = request.args.get("object", "")
    view = request.args.get("view", "023.png")
    if not valid_object(object_name):
        return jsonify(error="Unknown benchmark object."), 404
    record = record_for(object_name)
    if view not in record["input_views"] + record["heldout_views"]:
        return jsonify(error="Unknown view for this object."), 400
    render_path = Path(manifest_data()["dataset_dir"]) / record["render_dir"] / view
    if not render_path.is_file():
        return jsonify(error="Ground-truth image is unavailable at the configured dataset path."), 404
    return send_file(render_path)


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    method, object_name = payload.get("method"), payload.get("object")
    if method not in {"3dgs", "nerf", "neus"} or not isinstance(object_name, str) or not valid_object(object_name):
        return jsonify(error="Choose a valid method and object."), 400
    try:
        iterations = max(1, min(int(payload.get("iterations", 6000)), 1_000_000))
        image_size = max(32, min(int(payload.get("image_size", 256)), 2048))
    except (TypeError, ValueError):
        return jsonify(error="Iterations and image size must be whole numbers."), 400
    output = job_output(method, object_name)
    with lock:
        if any(j["status"] == "running" for j in jobs.values()):
            return jsonify(error="Only one GPU job is allowed at a time."), 409
        clear_previous_output(output)
        job_id = f"{method}-{object_name.replace('/', '_')}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        command, cwd = command_for(method, object_name, iterations, image_size, output)
        jobs[job_id] = {
            "id": job_id, "method": method, "object": object_name, "status": "running",
            "started": time.time(), "output": str(output), "log": str(output / "app.log"),
        }
        jobs[job_id]["output"] = str(output.relative_to(ASHRY))
        jobs[job_id]["log"] = str((output / "app.log").relative_to(ASHRY))
    threading.Thread(target=run_job, args=(job_id, command, cwd, output / "app.log"), daemon=True).start()
    return jsonify(jobs[job_id]), 202


@app.get("/api/jobs")
def job_list():
    with lock:
        result = []
        for job in jobs.values():
            item = dict(job)
            item["log_tail"] = read_tail(Path(job["log"]))
            result.append(item)
    return jsonify(result)


@app.post("/api/jobs/<job_id>/stop")
def stop(job_id: str):
    with lock:
        job = jobs.get(job_id)
        pid = job.get("pid") if job else None
    if not job:
        return jsonify(error="Job not found."), 404
    if job["status"] == "running" and pid:
        os.killpg(pid, signal.SIGTERM)
        with lock:
            job["status"] = "stopped"
    return jsonify(job)


@app.get("/api/jobs/<job_id>/artifacts")
def artifacts(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found."), 404
    output = job_output_path(job)
    files = [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()]
    return jsonify({"output": str(output.relative_to(ASHRY)), "files": files})


@app.post("/api/jobs/<job_id>/pointcloud")
def pointcloud(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found."), 404
    checkpoint = job_output_path(job) / "checkpoint.pt"
    if not checkpoint.is_file():
        return jsonify(error="checkpoint.pt is not available yet."), 409
    output = checkpoint.parent / "pointcloud.ply"
    try:
        points = export(
            job["method"],
            checkpoint,
            output,
            limit=4096 if job["method"] == "3dgs" else 50000,
        )
        metrics = compare(
            job["method"], output, MANIFEST, job["object"], checkpoint.parent / "pointcloud_metrics.json"
        )
    except (RuntimeError, ValueError, OSError, KeyError, MemoryError) as error:
        return jsonify(error=str(error)), 422
    return jsonify({"file": str(output.relative_to(ASHRY)), "points": len(points), "metrics": metrics})


@app.get("/api/jobs/<job_id>/textured-pointcloud")
def textured_pointcloud(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found."), 404
    if job["method"] != "3dgs":
        return jsonify(error="Textured point export is currently available for 3DGS only."), 400
    checkpoint = job_output_path(job) / "checkpoint.pt"
    if not checkpoint.is_file():
        return jsonify(error="checkpoint.pt is not available yet."), 409
    output = checkpoint.parent / "textured_pointcloud.ply"
    points = export_3dgs(checkpoint, output, 4096, colors_output=output)
    return jsonify({"file": str(output.relative_to(ASHRY)), "points": len(points)})


@app.get("/api/jobs/<job_id>/textured-mesh")
def textured_mesh(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found."), 404
    if job["method"] != "3dgs":
        return jsonify(error="Textured mesh extraction is currently available for 3DGS only."), 400
    checkpoint = job_output_path(job) / "checkpoint.pt"
    if not checkpoint.is_file():
        return jsonify(error="checkpoint.pt is not available yet."), 409
    try:
        return jsonify(export_3dgs_mesh(checkpoint))
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        return jsonify(error=str(error)), 422


@app.post("/api/objects/<path:object_name>/pointclouds")
def object_pointclouds(object_name: str):
    if not valid_object(object_name):
        return jsonify(error="Unknown benchmark object."), 404
    results = {}
    for method in ("3dgs", "nerf", "neus"):
        checkpoint = job_output(method, object_name) / "checkpoint.pt"
        if not checkpoint.is_file():
            results[method] = {"status": "missing", "message": "Train this model first."}
            continue
        output = checkpoint.parent / "pointcloud.ply"
        try:
            points = export(method, checkpoint, output, limit=4096 if method == "3dgs" else 50000)
            metrics = compare(
                method, output, MANIFEST, object_name, checkpoint.parent / "pointcloud_metrics.json"
            )
            results[method] = {
                "status": "ready",
                "file": str(output.relative_to(ASHRY)),
                "points": len(points),
                "metrics": metrics,
            }
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            results[method] = {"status": "failed", "message": str(error)}
    return jsonify({"object": object_name, "results": results})


@app.get("/artifacts/<path:relative_path>")
def artifact_file(relative_path: str):
    path = (ASHRY / relative_path).resolve()
    if ASHRY not in path.parents or not path.is_file():
        return jsonify(error="Artifact not found."), 404
    return send_from_directory(ASHRY, path.relative_to(ASHRY).as_posix())


@app.get("/")
def index():
    template = (Path(app.static_folder) / "index.html").read_text(encoding="utf-8")
    options = "".join(
        f'<option value="{record["category"]}/{record["object_id"]}">'
        f'{record["category"]}/{record["object_id"]} · {record["split"]}</option>'
        for record in manifest_data()["objects"]
    )
    template = template.replace(
        '<option>Loading objects…</option>',
        options or '<option>Unable to load objects</option>',
    )
    response = Response(template, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("ASHRY_PORT", "8000")), threaded=True)
