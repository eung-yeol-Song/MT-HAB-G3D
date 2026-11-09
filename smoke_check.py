"""Validate the artefacts of a --smoke run before launching the full matrix.
    python smoke_check.py outputs/smoke/runs
"""
import glob, json, os, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "outputs/smoke/runs"
ok = True
for d in sorted(glob.glob(os.path.join(root, "*_s0"))):
    tag = os.path.basename(d); mj = os.path.join(d, "metrics.json")
    if not os.path.exists(mj):
        print(f"[FAIL] {tag}: no metrics.json"); ok = False; continue
    m = json.load(open(mj)); z = np.load(os.path.join(d, "preds_test.npz"))
    checks = {
        "task recorded": m.get("task") in ("both", "cls", "reg"),
        "event_adj_h in preds": "event_adj_h" in z.files and z["event_adj_h"].ndim == 2,
        "D+4 positives ~ D+1 positives": abs(m["event_adjacent"]["D+4"]["n_pos"] - m["event_adjacent"]["D+1"]["n_pos"]) <= 5,
    }
    neural = m.get("n_params", 0) > 0
    if neural:
        checks["learned_params"] = isinstance(m.get("learned_params"), dict)
        checks["preds_val.npz"] = os.path.exists(os.path.join(d, "preds_val.npz"))
        checks["model_state.pt"] = os.path.exists(os.path.join(d, "model_state.pt"))
        if tag.startswith("A_mthab"):
            checks["permutation_importance"] = "permutation_importance" in m and len(m["permutation_importance"]["per_node"]) == 9
            checks["res_mask exported"] = any(k.endswith("res_mask") for k in m.get("learned_params", {}))
            checks["alpha_t exported"] = any(k.endswith("alpha_t") for k in m.get("learned_params", {}))
        if "cls_only" in tag:
            checks["cls-only: lambda=0"] = m["config"].get("task") == "cls"
        if "reg_only" in tag:
            checks["reg-only: task=reg"] = m["config"].get("task") == "reg"
    bad = [k for k, v in checks.items() if not v]
    print(f"[{'OK  ' if not bad else 'FAIL'}] {tag:24s} " + (", ".join(bad) if bad else f"{len(checks)} checks passed"))
    ok &= not bad
print("\nSMOKE CHECK:", "PASSED - launch the full matrix" if ok else "FAILED - fix before launching")
sys.exit(0 if ok else 1)
