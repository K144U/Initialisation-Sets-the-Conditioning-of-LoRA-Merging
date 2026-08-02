import json, os
RES = "results/phase3"
def w(p):
    p = os.path.join(RES, p)
    return json.load(open(p))["worst_task_excess"] if os.path.exists(p) else None
import statistics
SEEDS = ["seed1","seed2","seed3"]
LAM = {"llama31_8b":"0p05","mistral_7b":"0p13","qwen25_7b":"0p13","yi15_9b":"0p13"}
ALPHAS = [("0p1",.10),("0p15",.15),("0p25",.25),("0p35",.35),("0p5",.50),("0p75",.75),("1",1.0)]

print("W1 VERDICT: rd-encoder ridge vs a merge-coefficient-tuned Task Arithmetic\n")
print(f"{'base':<13}{'TA@.25':>9}{'bestTA':>9}{'a*':>6}{'rd s1':>9}{'rd 3seed':>10}"
      f"{'TIES':>9}{'gap s1':>9}{'gap 3s':>9}")
rows={}
for b,lam in LAM.items():
    al={a:w(f"eval_w1_alpha/{b}__ta_alpha{t}__seed1.json") for t,a in ALPHAS}
    al={k:v for k,v in al.items() if v is not None}
    rd1 = w(f"eval_seed_rdridge_regmean/{b}__rd_ridge__seed1.json") or w(f"eval_ridge_seed/{b}__ridge_l{lam}__seed1.json")
    rd3 = [w(f"eval_seed_rdridge_regmean/{b}__rd_ridge__{s}.json") or w(f"eval_ridge_seed/{b}__ridge_l{lam}__{s}.json") for s in SEEDS]
    rd3 = statistics.mean([x for x in rd3 if x is not None])
    ties = statistics.mean([w(f"eval_matrix_seeds/{b}__ties__{s}.json") for s in SEEDS])
    ba=min(al,key=al.get); bv=al[ba]
    rows[b]=(al,ba,bv,rd1,rd3,ties)
    print(f"{b:<13}{al[0.25]:>9.4f}{bv:>9.4f}{ba:>6.2f}{rd1:>9.4f}{rd3:>10.4f}"
          f"{ties:>9.4f}{bv-rd1:>+9.4f}{bv-rd3:>+9.4f}")

print("\n--- what the paper claims vs what survives a tuned TA ---")
print(f"{'base':<13}{'paper: rd vs TA@.25':>22}{'honest: rd vs bestTA':>23}")
for b,(al,ba,bv,rd1,rd3,ties) in rows.items():
    old = 100*(1-rd3/al[0.25]); new = 100*(1-rd3/bv)
    print(f"{b:<13}{old:>21.0f}%{new:>22.0f}%")

print("\n--- does 'only TIES separates from the structure-blind cluster' hold? ---")
for b,(al,ba,bv,rd1,rd3,ties) in rows.items():
    print(f"  {b:<13} tunedTA {bv:.4f}  vs TIES {ties:.4f}  -> "
          + ("tuned TA BEATS TIES" if bv < ties else "TIES still ahead"))
