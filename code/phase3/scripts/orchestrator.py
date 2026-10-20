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
STATE = ROOT / "logs" / os.environ.get("ORCH_STATE", "orchestrator_state.json")
BACKOFF_S = 600
VRAM_POLL_TRIES = 3


# Users whose jobs must never share a card with ours. A GPU running any of
# their processes counts as having no free memory, whatever nvidia-smi says,
# because their jobs may grow into memory that looks free now.
PROTECT_USERS = {u for u in os.environ.get("PROTECT_USERS", "").split(",") if u}


def protected_user_on(gpu: int) -> str | None:
    if not PROTECT_USERS:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
             "-i", str(gpu)], capture_output=True, text=True, timeout=30)
        for pid in out.stdout.split():
            who = subprocess.run(["ps", "-o", "user:32=", "-p", pid.strip()],
                                 capture_output=True, text=True, timeout=10)
            user = who.stdout.strip()
            if user in PROTECT_USERS:
                return user
    except Exception:
        return "unknown (query failed)"
    return None


# Dynamic mode shares cards with other users' jobs. HEADROOM_GB is kept free
# beyond a cell's own need so their jobs can still grow, and a claim file per
# GPU keeps two of OUR jobs from ever landing on the same card.
HEADROOM_GB = float(os.environ.get("ORCH_HEADROOM_GB", "10"))
CLAIMS = ROOT / "logs" / "gpu_claims"
# 2026-10-19: a claim is dropped as soon as the process that wrote it is gone
# (a qdel'd or walltime-killed job left gpu0 claimed for hours). GPU indices
# already assume one node, so the pid is checked on this one. The age limit is
# only the fallback for an unreadable claim or a reused pid, and is longer
# than any cell (a 4-epoch training cell runs ~10 h).
CLAIM_STALE_S = 48 * 3600


def _claim_path(gpu) -> Path:
    return CLAIMS / f"gpu{gpu}"


def _claimer_alive(p: Path) -> bool:
    try:
        pid = int(p.read_text().split()[1])
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError, IndexError):
        return True        # unreadable, or a pid we may not signal: assume alive
    return True


def claimed_by_other(gpu) -> bool:
    p = _claim_path(gpu)
    if not p.exists():
        return False
    try:
        if not _claimer_alive(p) or time.time() - p.stat().st_mtime > CLAIM_STALE_S:
            p.unlink()
            return False
    except OSError:
        return True
    return True


def claim(gpu) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(_claim_path(gpu), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('PBS_JOBID', 'nojob')} {os.getpid()}\n".encode())
    os.close(fd)
    return True


def release(gpu) -> None:
    try:
        _claim_path(gpu).unlink()
    except OSError:
        pass


# 2026-10-21: a job whose ORCH_STATE is listed in this file (one name per line)
# does not sleep out the backoff when no card is free. It watches the claim
# files instead and takes a card the moment another of our jobs releases one,
# keeping it only if the card has need + HEADROOM_GB free. Every other job
# relaunches straight after its cell finishes, so a job that backs off for
# 10 min almost never gets a card (the regime trainings got none for 17 h).
# The file is read at every wait, so priorities change without a resubmit.
PRIORITY_FILE = ROOT / "logs" / "orch_priority"
WATCH_POLL_S = 0.25
WATCH_RETRY_S = 60


def is_priority() -> bool:
    try:
        return STATE.name in PRIORITY_FILE.read_text().split()
    except OSError:
        return False


def free_gb(gpu: int) -> float:
    if protected_user_on(gpu):
        return 0.0
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

    def pick_gpu(self, need: float):
        """Dynamic mode: the allowed, unclaimed GPU with the most free memory
        above need + HEADROOM_GB, claimed atomically; or None."""
        best, best_free = None, -1.0
        for g in self.candidates:
            if claimed_by_other(g):
                continue
            f = free_gb(g)
            if f >= need + HEADROOM_GB and f > best_free:
                best, best_free = g, f
        if best is not None and not claim(best):
            return None
        return best

    def watch_gpu(self, need: float, seconds: float):
        """Priority jobs: claim a card as soon as its claim file goes away (the
        releasing job's own next pick runs nvidia-smi on every free card
        first), keep it if it has need + HEADROOM_GB free, else release it
        and leave that card alone for WATCH_RETRY_S. If another unclaimed card
        is freer by then, it moves there, as pick_gpu would have chosen.
        None after `seconds`."""
        tried = {}
        end = time.time() + seconds
        while time.time() < end:
            for g in self.candidates:
                if time.time() - tried.get(g, 0.0) < WATCH_RETRY_S:
                    continue
                if claimed_by_other(g) or not claim(g):
                    continue
                f = free_gb(g)
                if f < need + HEADROOM_GB:
                    release(g)
                    tried[g] = time.time()
                    continue
                for h in self.candidates:
                    if h != g and not claimed_by_other(h):
                        fh = free_gb(h)
                        if fh > f and claim(h):
                            release(g)
                            g, f = h, fh
                return g
            time.sleep(WATCH_POLL_S)
        return None

    def worker(self, gpu):
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
            prio = self.dynamic and is_priority()
            for _ in range(VRAM_POLL_TRIES):
                if self.dynamic:
                    gpu = self.pick_gpu(need)
                    if gpu is not None:
                        ok = True
                        break
                elif free_gb(gpu) >= need:
                    ok = True
                    break
                if prio:
                    gpu = self.watch_gpu(need, BACKOFF_S)
                    ok = gpu is not None
                    break
                time.sleep(60)
            if not ok:
                print(f"[gpu{gpu}] {name}: <{need}GB free, requeue + backoff",
                      flush=True)
                self.q.put(cell)
                self.write_state()
                if not prio:       # a priority job has just waited BACKOFF_S
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
            if self.dynamic:
                release(gpu)
            gate_backoff = False
            with self.lock:
                self.running.pop(name, None)
                if rc == 0 and (Path(cell["done"]).exists()
                                or (ROOT / cell["done"]).exists() or self.dry):
                    self.done.append(name)
                    print(f"[gpu{gpu}] DONE {name} ({dt:.0f} min)", flush=True)
                elif rc == 87:
                    # Inner VRAM gate refused this GPU (boundary flicker on a
                    # shared card) — not the cell's fault. Requeue WITHOUT
                    # charging an attempt, then back off like the poll path, so
                    # a borderline GPU can never burn a cell's retry budget.
                    print(f"[gpu{gpu}] GATE {name} rc=87 ({dt:.0f} min) "
                          f"— no-charge requeue + backoff", flush=True)
                    self.q.put(cell)
                    gate_backoff = True
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
            if gate_backoff:
                time.sleep(BACKOFF_S)

    def run(self):
        # ORCH_DYNAMIC=1: one worker, which picks a GPU from self.gpus per cell.
        # For queues where ngpus is accounting only and no device is bound:
        # the job then occupies exactly one GPU at a time, as it was billed.
        self.dynamic = os.environ.get("ORCH_DYNAMIC") == "1"
        self.candidates = list(self.gpus)
        # ORCH_WORKERS: concurrent cells in dynamic mode; must not exceed the
        # job's billed ngpus.
        slots = ([None] * int(os.environ.get("ORCH_WORKERS", "1")) if self.dynamic
                 else self.gpus)
        threads = [threading.Thread(target=self.worker, args=(g,), daemon=True)
                   for g in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.write_state()
        sentinel = os.environ.get("ORCH_SENTINEL", "_QUEUE_COMPLETE")
        if not self.failed:
            (ROOT / sentinel).write_text(time.strftime("%F %T\n"))
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
    # ORCH_SHARD="i/N": this job takes cells whose index is i mod N, so N jobs
    # can split one manifest without two of them ever running the same cell.
    shard = os.environ.get("ORCH_SHARD")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]
        print(f"[orch] shard {i}/{n}: {len(cells)} cells", flush=True)
    # GPU tokens are passed through as strings: under PBS ngpus allocation
    # CUDA_VISIBLE_DEVICES may hold a UUID rather than an index.
    gpus = [g.strip() for g in os.environ.get("GPUS", "0,1,2,3,5").split(",") if g.strip()]
    print(f"[orch] GPUs: {gpus}; manifest: {args.manifest} "
          f"({len(cells)} cells)", flush=True)
    Orchestrator(cells, gpus, args.dry_run).run()


if __name__ == "__main__":
    main()
