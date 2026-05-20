"""Tiny PBS probe — verifies torch sees the GPU and bf16 works.

Submitted via pbs_cuda_probe.sh as a 1-min job. Prints a summary
and exits 0 on success, non-zero on failure.
"""

import json
import os
import sys
import time

ok = True
report: dict = {
    "hostname": os.environ.get("HOSTNAME", ""),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "pbs_jobid": os.environ.get("PBS_JOBID", ""),
}

try:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        report["device_name"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info(0)
        report["vram_free_gb"] = round(free / (1024 ** 3), 2)
        report["vram_total_gb"] = round(total / (1024 ** 3), 2)
        # bf16 sanity
        x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
        t0 = time.time()
        y = x @ x
        torch.cuda.synchronize()
        report["bf16_matmul_2048_seconds"] = round(time.time() - t0, 4)
        report["bf16_result_finite"] = bool(torch.isfinite(y).all().item())
    else:
        ok = False
        report["error"] = "cuda not available"
except Exception as e:
    ok = False
    report["error"] = f"{type(e).__name__}: {e}"

print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if ok else 1)
