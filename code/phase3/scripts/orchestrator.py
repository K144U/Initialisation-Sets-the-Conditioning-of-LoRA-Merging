"""Multi-GPU work-queue orchestrator for rdmerge (master plan I0 + E-campaign).

One PBS job drives one worker per GPU; cells are shell commands with a
done-file for idempotency. Modeled on the LMCA parallel-grid orchestrator
that completed a 1,020-cell campaign on this cluster.

  python orchestrator.py --manifest <cells.json> [--dry-run]

Manifest: JSON list of cells, each
  {"name": str, "cmd": str, "done": str (path), "min_free_gb": float}
Cells run with CUDA_VISIBLE_DEVICES pinned to the worker's GPU. A cell is
skipped if its done-file exists (resume-safety across requeues). If the
worker's GPU lacks min_free_gb of free VRAM (shared node), the cell is
requeued and the worker backs off — it does not steal another worker's GPU.
Failed cells are retried once, then parked in the failed list.

GPUs: env GPUS="0,1,2,3,5" (set by the PBS wrapper, which reads _ORCH_GPUS
at the project root if present — edit that file + requeue to change the
GPU set without touching code).

State: logs/orchestrator_state.json (pending/running/done/failed) refreshed
every cycle; per-cell logs in logs/orch/<cell>.log. When the queue drains
with no failures the sentinel _QUEUE_COMPLETE is written (the keeper stops
requeueing on it); with failures, _QUEUE_FAILED listing them.
"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGDIR = ROOT / "logs" / "orch"
STATE = ROOT / "logs" / "orchestrator_state.json"
BACKOFF_S = 600
VRAM_POLL_TRIES = 3


def free_gb(gpu: int) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) / 1024.0
    except Exception:
        return 0.0


class Orchestrator:
    def __init__(self, cells, gpus, dry):
        self.q = queue.Queue()
        self.gpus = gpus
        self.dry = dry
        self.lock = threading.Lock()
        self.done, self.failed, self.running = [], [], {}
        self.attempts = {}
        self.skipped = []
        for c in cells:
            if Path(c["done"]).exists() or (ROOT / c["done"]).exists():
                self.skipped.append(c["name"])
            else:
                self.q.put(c)
        print(f"[orch] {self.q.qsize()} pending, {len(self.skipped)} already done",
              flush=True)

    def write_state(self):
        with self.lock:
            state = {
                "ts": time.strftime("%F %T"),
                "pending": self.q.qsize(),
                "running": dict(self.running),
                "done": self.done + self.skipped,
                "failed": self.failed,
            }
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1))

    def worker(self, gpu: int):
        while True:
            try:
                cell = self.q.get(timeout=30)
            except queue.Empty:
                return
            name = cell["name"]
            if Path(cell["done"]).exists() or (ROOT / cell["done"]).exists():
                with self.lock:
                    self.done.append(name)
                continue
            need = float(cell.get("min_free_gb", 30))
            ok = False
            for _ in range(VRAM_POLL_TRIES):
                if free_gb(gpu) >= need:
                    ok = True
                    break
                time.sleep(60)
            if not ok:
                print(f"[gpu{gpu}] {name}: <{need}GB free, requeue + backoff",
                      flush=True)
                self.q.put(cell)
                self.write_state()
                time.sleep(BACKOFF_S)
                continue
            with self.lock:
                self.running[name] = gpu
            self.write_state()
            print(f"[gpu{gpu}] LAUNCH {name}", flush=True)
            t0 = time.time()
            if self.dry:
                rc = 0
            else:
                env = dict(os.environ,
                           CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1",
                           TOKENIZERS_PARALLELISM="false")
                LOGDIR.mkdir(parents=True, exist_ok=True)
                with open(LOGDIR / f"{name}.log", "a") as lf:
                    rc = subprocess.run(
                        cell["cmd"], shell=True, cwd=ROOT, env=env,
                        stdout=lf, stderr=subprocess.STDOUT).returncode
            dt = (time.time() - t0) / 60
            with self.lock:
                self.running.pop(name, None)
                if rc == 0 and (Path(cell["done"]).exists()
                                or (ROOT / cell["done"]).exists() or self.dry):
                    self.done.append(name)
                    print(f"[gpu{gpu}] DONE {name} ({dt:.0f} min)", flush=True)
                else:
                    n = self.attempts.get(name, 0) + 1
                    self.attempts[name] = n
                    if n < 2:
                        print(f"[gpu{gpu}] FAIL {name} rc={rc} ({dt:.0f} min) "
                              f"— retry queued", flush=True)
                        self.q.put(cell)
                    else:
                        self.failed.append(name)
                        print(f"[gpu{gpu}] FAIL {name} rc={rc} — parked",
                              flush=True)
            self.write_state()

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(g,), daemon=True)
                   for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.write_state()
        if not self.failed:
            (ROOT / "_QUEUE_COMPLETE").write_text(time.strftime("%F %T\n"))
            print("[orch] queue complete", flush=True)
        else:
            (ROOT / "_QUEUE_FAILED").write_text("\n".join(self.failed) + "\n")
            print(f"[orch] finished with {len(self.failed)} failed cells",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cells = json.loads(Path(args.manifest).read_text())
    gpus = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,5").split(",")]
    print(f"[orch] GPUs: {gpus}; manifest: {args.manifest} "
          f"({len(cells)} cells)", flush=True)
    Orchestrator(cells, gpus, args.dry_run).run()


if __name__ == "__main__":
    main()
