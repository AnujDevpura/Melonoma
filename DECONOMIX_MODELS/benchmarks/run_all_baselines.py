"""
Run actual baseline implementations with progress and unified scoring.

Includes:
 - OLS / NNLS (Python, fast) for sanity.
 - MuSiC (R): via benchmarks/R/run_music.R (requires R + packages).
 - BayesPrism (R): via benchmarks/R/run_bayesprism.R.
 - CIBERSORT (R): wrapper expects official CIBERSORT.R and a signature CSV.
 - (Optional) Scaden (CLI): if installed, can be wired similarly.

This script does NOT modify Data/prepared. It writes temporary CSVs and
final metrics under --out.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import time as _time


def _read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_prepared(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    cell_types = _read_lines(data_dir / "cell_types.txt")
    genes = _read_lines(data_dir / "selected_genes.txt") if (data_dir / "selected_genes.txt").exists() else [f"g{i}" for i in range(X_train.shape[1])]
    return X_train, y_train, X_val, y_val, X_test, y_test, cell_types, genes


def export_bulk_csv(X: np.ndarray, genes: List[str], out_csv: Path) -> None:
    df = pd.DataFrame(X, columns=genes)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def project_to_simplex(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    s = y.sum(axis=1, keepdims=True)
    s = np.where(s < eps, eps, s)
    return y / s


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, cell_types: List[str]) -> pd.DataFrame:
    y_pred = project_to_simplex(y_pred)
    K = y_true.shape[1]
    rhos = np.zeros(K)
    for k in range(K):
        try:
            r, _ = spearmanr(y_true[:, k], y_pred[:, k])
            if np.isnan(r):
                r = 0.0
        except Exception:
            r = 0.0
        rhos[k] = r
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    df = pd.DataFrame({
        "cell_type": cell_types,
        "spearman": rhos,
        "mae": mae,
        "avg_prop": y_true.mean(axis=0),
    })
    avg = pd.DataFrame({
        "cell_type": ["AVERAGE"],
        "spearman": [float(np.nanmean(rhos))],
        "mae": [float(np.mean(mae))],
        "avg_prop": [float(y_true.mean())],
    })
    return pd.concat([df, avg], ignore_index=True)


def run_python_ols(Xtr, ytr, Xev) -> np.ndarray:
    model = LinearRegression(n_jobs=None)
    model.fit(Xtr, ytr)
    return model.predict(Xev)


def run_python_nnls(Xtr, ytr, Xev, desc: str = "NNLS") -> np.ndarray:
    # Learn signature S from train via least squares, then NNLS per sample.
    S, *_ = np.linalg.lstsq(ytr, Xtr, rcond=None)
    S = np.clip(S, 0.0, None)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-12)
    ST = S.T
    W = np.zeros((Xev.shape[0], S.shape[0]))
    with tqdm(total=Xev.shape[0], desc=desc, leave=False) as pbar:
        for i in range(Xev.shape[0]):
            w, _ = nnls(ST, Xev[i].astype(np.float64, copy=False))
            W[i] = w
            if (i + 1) % 10 == 0 or i + 1 == Xev.shape[0]:
                pbar.update(min(10, Xev.shape[0] - i))
    return W


def check_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def run_with_progress(cmd: List[str], desc: str, cwd: Path | None = None) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)
    with tqdm(total=None, desc=desc, leave=False) as pbar:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            _time.sleep(0.2)
            pbar.update(0.2)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Run actual baselines with progress")
    ap.add_argument("--data-dir", default="DECONOMIX_MODELS/Data/prepared")
    ap.add_argument("--h5ad", default="DECONOMIX_MODELS/Data/rna_data.h5ad", help="Path to single-cell atlas for R baselines")
    ap.add_argument("--out", default="DECONOMIX_MODELS/results/AllBaselines")
    ap.add_argument("--models", nargs="+", default=["ols", "nnls", "music", "bayesprism" , "cibersort"])  # add scaden when ready
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--cibersort-r", default=None, help="Path to official CIBERSORT.R (required for cibersort)")
    ap.add_argument("--signature", default=None, help="Signature CSV for cibersort (genes as columns)")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xv, yv, Xte, yte, cell_types, genes = load_prepared(data_dir)
    Xev, yev = (Xte, yte) if args.split == "test" else (Xv, yv)

    # Export evaluation bulk expression to CSV (samples x genes)
    bulk_csv = out_root / f"bulk_{args.split}_expr.csv"
    export_bulk_csv(Xev, genes, bulk_csv)

    summary_rows: List[Dict[str, float]] = []

    def finish_model(name: str, yhat: np.ndarray | None, preds_csv: Path) -> None:
        if yhat is None:
            if preds_csv.exists():
                dfp = pd.read_csv(preds_csv)
                # Expect columns in cell_types order
                yhat_np = dfp[cell_types].values if all(c in dfp.columns for c in cell_types) else dfp.values
            else:
                print(f"[Skip] {name}: no predictions")
                return
        else:
            yhat_np = yhat
            pd.DataFrame(yhat_np, columns=cell_types).to_csv(preds_csv, index=False)
        perf = evaluate(yev, yhat_np, cell_types)
        per_csv = out_root / f"performance_{name}.csv"
        perf.to_csv(per_csv, index=False)
        avg = perf.loc[perf["cell_type"] == "AVERAGE"].iloc[0]
        print(f"  -> {name}: Avg Spearman={avg['spearman']:.3f} MAE={avg['mae']:.4f}  saved={per_csv}")
        summary_rows.append({"model": name, "spearman": float(avg["spearman"]), "mae": float(avg["mae"])})

    print("=" * 70)
    print("RUNNING BASELINES")
    print("=" * 70)
    print(f"Data: {data_dir}")
    print(f"Split: {args.split} -> {Xev.shape}")
    print("Models:", ", ".join(args.models))

    for m in args.models:
        print(f"\n[Run] {m.upper()} ...")
        t0 = time.time()
        preds_csv = out_root / f"pred_props_{m}.csv"

        if m == "ols":
            with tqdm(total=1, desc="OLS", leave=False) as pbar:
                yhat = run_python_ols(Xtr, ytr, Xev)
                pbar.update(1)
            finish_model(m, yhat, preds_csv)
        elif m == "nnls":
            yhat = run_python_nnls(Xtr, ytr, Xev, desc="NNLS samples")
            finish_model(m, yhat, preds_csv)
        elif m == "music":
            if not check_cmd(args.rscript):
                print("  -> Rscript not found; skipping MuSiC")
                continue
            script = Path("DECONOMIX_MODELS/benchmarks/R/run_music.R")
            cmd = [args.rscript, str(script),
                   f"--h5ad={args.h5ad}",
                   f"--bulk={bulk_csv}",
                   f"--genes={'|'.join(genes)}",
                   f"--out={preds_csv}"]
            rc = run_with_progress(cmd, desc="MuSiC (R)")
            if rc != 0:
                print("  -> MuSiC failed; see R output")
                continue
            finish_model(m, None, preds_csv)
        elif m == "bayesprism":
            if not check_cmd(args.rscript):
                print("  -> Rscript not found; skipping BayesPrism")
                continue
            script = Path("DECONOMIX_MODELS/benchmarks/R/run_bayesprism.R")
            cmd = [args.rscript, str(script),
                   f"--h5ad={args.h5ad}",
                   f"--bulk={bulk_csv}",
                   f"--genes={'|'.join(genes)}",
                   f"--out={preds_csv}"]
            rc = run_with_progress(cmd, desc="BayesPrism (R)")
            if rc != 0:
                print("  -> BayesPrism failed; see R output")
                continue
            finish_model(m, None, preds_csv)
        elif m == "cibersort":
            if not check_cmd(args.rscript):
                print("  -> Rscript not found; skipping CIBERSORT")
                continue
            if not args.cibersort_r or not args.signature:
                print("  -> Provide --cibersort-r and --signature to run CIBERSORT")
                continue
            script = Path("DECONOMIX_MODELS/benchmarks/R/run_cibersort.R")
            cmd = [args.rscript, str(script),
                   f"--cibersort={args.cibersort_r}",
                   f"--signature={args.signature}",
                   f"--mixture={bulk_csv}",
                   f"--out={preds_csv}"]
            rc = run_with_progress(cmd, desc="CIBERSORT (R)")
            if rc != 0:
                print("  -> CIBERSORT failed; see R output")
                continue
            finish_model(m, None, preds_csv)
        else:
            print("  -> Unknown model key; skipping")

        dt = time.time() - t0
        print(f"  [Done] {m} in {dt:.1f}s")

    summary = pd.DataFrame(summary_rows).sort_values(by="spearman", ascending=False)
    summary_csv = out_root / "summary_all_baselines.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n[Summary] {summary_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
