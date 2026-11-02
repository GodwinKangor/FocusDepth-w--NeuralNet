"""Live presentation mode: AirDrop a capture to the Mac and the results appear in the browser.

    python python/demo.py                 # watches ~/Downloads, serves http://localhost:8765
    python python/demo.py --watch DIR --port 8765 --no-browser

Watches a folder for new capture zips (the iOS app's "Share capture (.zip)"),
unpacks each into data/raw/<name>/, runs run_pipeline.py on it, and serves a
full-screen page that shows the focus sweep while processing and the depth
maps when done. Scenes already in results/ can be browsed from the same page,
so the demo still works if AirDrop or the venue lighting fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

log = logging.getLogger("focusdepth.demo")
REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
RESULTS = REPO / "results"
PAGE = Path(__file__).resolve().parent / "demo_page.html"

# Pipeline log lines that mark the end of each stage shown on the page.
STAGES = [
    ("unpack", "Unpacking capture", None),
    ("register", "Aligning focus frames", "registered"),
    ("optical", "Depth from focus + confidence", "optical depth and confidence done"),
    ("neural", "Neural depth (Depth Anything V2)", "neural depth done"),
    ("fusion", "Confidence-guided fusion", "wrote results"),
]
RESULT_FILES = ["reference_rgb.png", "optical_depth.png", "neural_depth.png", "fixed_fusion.png", "confidence_fusion.png", "optical_confidence.png"]


class State:
    """Shared between the watcher thread and the HTTP handler; served as /api/state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "waiting"          # waiting | processing | done | error
        self.current: str | None = None
        self.stage = 0
        self.error: str | None = None
        self.started = 0.0
        self.n_frames = 0

    def update(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "current": self.current,
                "stage": self.stage,
                "n_frames": self.n_frames,
                "stages": [label for _, label, _ in STAGES],
                "error": self.error,
                "elapsed_s": round(time.time() - self.started, 1) if self.started else 0,
                "scenes": list_scenes(),
            }


def list_scenes() -> list[dict]:
    """Processed scenes, newest first."""
    scenes = []
    for d in RESULTS.iterdir() if RESULTS.exists() else []:
        if not (d / "confidence_fusion.png").exists() or not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        scenes.append({
            "name": d.name,
            "mtime": (d / "metrics.json").stat().st_mtime,
            "frames": sorted(p.name for p in (d / "demo_frames").glob("*.jpg")) if (d / "demo_frames").exists() else [],
            "lens": m.get("lens_positions", []),
            "confidence_mean": m.get("confidence", {}).get("mean"),
            "device": m.get("neural", {}).get("device"),
            "timings_s": m.get("timings_s", {}),
            "ordering": {k: v.get("accuracy") for k, v in m.get("evaluation", {}).get("ordering", {}).items()},
        })
    return sorted(scenes, key=lambda s: s["mtime"], reverse=True)


def safe_extract(zip_path: Path, dest_root: Path) -> Path:
    """Extract a capture zip (one top-level folder holding manifest.json) and return that folder."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1 or f"{next(iter(tops))}/manifest.json" not in names:
            raise ValueError(f"{zip_path.name} is not a FocusDepth capture (expected <folder>/manifest.json)")
        top = next(iter(tops))
        for n in names:
            target = (dest_root / n).resolve()
            if dest_root.resolve() not in target.parents and target != dest_root.resolve():
                raise ValueError(f"unsafe path in zip: {n}")
            zf.extract(n, dest_root)
    return dest_root / top


def write_frame_thumbnails(scene_dir: Path, out_dir: Path, height: int = 720) -> None:
    """Small copies of the focus frames (sorted by lens position) for the sweep animation."""
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    frames = sorted(manifest["frames"], key=lambda f: f.get("lens_position_actual", f.get("lens_position", 0)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        img = cv2.imread(str(scene_dir / f["file"]))
        if img is None:
            continue
        scale = height / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def process(zip_path: Path, state: State, model: str) -> None:
    state.update(status="processing", current=zip_path.stem, stage=0, error=None, started=time.time())
    try:
        scene_dir = safe_extract(zip_path, RAW)
        name = scene_dir.name
        out = RESULTS / name
        write_frame_thumbnails(scene_dir, out / "demo_frames")
        state.update(current=name, stage=1, n_frames=len(list((out / "demo_frames").glob("*.jpg"))))
        cmd = [sys.executable, str(REPO / "python" / "run_pipeline.py"), "--input", str(scene_dir), "--output", str(out), "--model", model, "--fusion", "all"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            log.info("  %s", line)
            tail = (tail + [line])[-15:]
            for i, (_, _, marker) in enumerate(STAGES):
                if marker and marker in line:
                    state.update(stage=i + 1)
        if proc.wait() != 0:
            raise RuntimeError("pipeline failed:\n" + "\n".join(tail))
        state.update(status="done", stage=len(STAGES))
        log.info("done: %s (%.1f s)", name, time.time() - state.started)
    except Exception as exc:  # shown on the page; the watcher keeps running
        log.error("%s", exc)
        state.update(status="error", error=str(exc))


def watch(folder: Path, state: State, model: str, poll_s: float = 1.0) -> None:
    """Process zips that appear in `folder` after startup, once their size stops changing."""
    seen = {p.name for p in folder.glob("*.zip")}
    sizes: dict[str, int] = {}
    log.info("watching %s for capture zips", folder)
    while True:
        for p in sorted(folder.glob("*.zip"), key=lambda p: p.stat().st_mtime):
            if p.name in seen:
                continue
            size = p.stat().st_size
            if sizes.get(p.name) != size:  # still arriving
                sizes[p.name] = size
                continue
            seen.add(p.name)
            log.info("new capture: %s", p.name)
            process(p, state, model)
        time.sleep(poll_s)


class Handler(SimpleHTTPRequestHandler):
    """Serves the page, /api/state, and files under results/."""

    def __init__(self, *args, state: State, **kwargs):
        self.state = state
        super().__init__(*args, directory=str(RESULTS), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/api/state"):
            return self._send(json.dumps(self.state.snapshot()).encode(), "application/json")
        if self.path.startswith("/results/"):
            self.path = self.path[len("/results"):]
            return super().do_GET()
        self.send_error(404)

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # keep the terminal readable
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=Path, default=Path.home() / "Downloads", help="folder AirDrop saves into")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="depth-anything-v2-small")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    RESULTS.mkdir(exist_ok=True)
    state = State()
    threading.Thread(target=watch, args=(args.watch, state, args.model), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, state=state))
    url = f"http://localhost:{args.port}/"
    log.info("presenter page: %s  (Ctrl-C to stop)", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
