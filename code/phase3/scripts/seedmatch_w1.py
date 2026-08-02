import json, os
RES = "results/phase3"

def w(p):
    p = os.path.join(RES, p)
    return json.load(open(p))["worst_task_excess"] if os.path.exists(p) else None

ALPHAS = [("0p1", 0.10), ("0p15", 0.15), ("0p25", 0.25), ("0p35", 0.35),
          ("0p5", 0.50), ("0p75", 0.75), ("1", 1.00)]
LAM = {"llama31_8b": "0p05", "mistral_7b": "0p13",
       "qwen25_7b": "0p13", "yi15_9b": "0p13"}

print("SEED-1 MATCHED (every number below is seed1, identical adapters)\n")
print(f"{'base':<13}{'TA a=.25':>10}{'best TA':>10}{'at a':>7}"
      f"{'rd-ridge':>10}{'TIES':>9}{'gap':>9}  reading")
for b, lam in LAM.items():
    alphas = {}
    for tag, a in ALPHAS:
        v = w(f"eval_w1_alpha/{b}__ta_alpha{tag}__seed1.json")
        if v is not None:
            alphas[a] = v
    rd = (w(f"eval_seed_rdridge_regmean/{b}__rd_ridge__seed1.json")
          or w(f"eval_ridge_seed/{b}__ridge_l{lam}__seed1.json"))
    ties = w(f"eval_matrix_seeds/{b}__ties__seed1.json")
    if not alphas or rd is None:
        print(f"{b:<13}{'pending':>10}")
        continue
    ba = min(alphas, key=alphas.get)
    bv = alphas[ba]
    gap = bv - rd
    if gap < -0.005:
        rd_read = "tuned TA WINS"
    elif gap <= 0.005:
        rd_read = "TIE (seed noise)"
    else:
        rd_read = "rd-ridge wins"
    d25 = alphas.get(0.25)
    print(f"{b:<13}{(d25 if d25 else float('nan')):>10.4f}{bv:>10.4f}{ba:>7.2f}"
          f"{rd:>10.4f}{ties:>9.4f}{gap:>+9.4f}  {rd_read}")
    print(f"{'':<13}sweep: " + "  ".join(f"{a}={alphas[a]:.4f}" for a in sorted(alphas)))
print("\ngap = bestTA - rd-ridge; negative means a tuned scalar on TA beats the paper's method.")
