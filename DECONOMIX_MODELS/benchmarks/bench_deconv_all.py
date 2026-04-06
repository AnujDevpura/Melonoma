"""
Extended baseline benchmarking for deconvolution on prepared arrays.

Runs multiple baselines on the same train/val/test arrays produced by
prepare_deconv_data.py and reports per–cell-type Spearman/MAE plus an
AVERAGE row. Does not modify Data/prepared.

Included baselines (Python-only):
 - ols: multi-output linear regression (unconstrained)
 - ridge: multi-output ridge with CV over alphas
 - nnls: non-negative linear regression per output
 - mean: predicts mean training proportions for all samples
 - cibersort_like: NNLS against a signature S; if --signature is not given,
   S is learned from (X_train, y_train) via least squares and row-normalized
 - music_like: weighted-NNLS using gene inverse-variance (bulk-only approx)
 - bayesprism_like: entropy-regularized deconvolution via projected gradient
 - scaden_like: MLPRegressor trained on (X_train, y_train)
 - dtd_like: signature learned from train + NNLS
 - adtd_like: ridge-refined signature anchored to initial + NNLS

Usage
  python DECONOMIX_MODELS/benchmarks/bench_deconv_all.py \
      --data-dir DECONOMIX_MODELS/Data/prepared \
      --models ols ridge nnls mean cibersort_like music_like bayesprism_like scaden_like dtd_like adtd_like \
      --out DECONOMIX_MODELS/results/Baselines_run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


def _read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_prepared(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    cell_types = _read_lines(data_dir / "cell_types.txt")
    return X_train, y_train, X_val, y_val, X_test, y_test, cell_types


def maybe_read_genes(data_dir: Path) -> List[str] | None:
    p = data_dir / "selected_genes.txt"
    if p.exists():
        return _read_lines(p)
    return None


def project_to_simplex(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    s = y.sum(axis=1, keepdims=True)
    s = np.where(s < eps, eps, s)
    return y / s


def spearman_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    K = y_true.shape[1]
    out = np.zeros(K, dtype=float)
    for k in range(K):
        try:
            r, _ = spearmanr(y_true[:, k], y_pred[:, k])
            if np.isnan(r):
                r = 0.0
        except Exception:
            r = 0.0
        out[k] = r
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, cell_types: List[str]) -> pd.DataFrame:
    y_pred = project_to_simplex(y_pred)
    spearman = spearman_per_column(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    df = pd.DataFrame({
        "cell_type": cell_types,
        "spearman": spearman,
        "mae": mae,
        "avg_prop": y_true.mean(axis=0),
    })
    avg_row = pd.DataFrame({
        "cell_type": ["AVERAGE"],
        "spearman": [float(np.nanmean(spearman))],
        "mae": [float(np.mean(mae))],
        "avg_prop": [float(y_true.mean())],
    })
    return pd.concat([df, avg_row], ignore_index=True)


def run_ols(X_train, y_train, X_eval) -> np.ndarray:
    model = LinearRegression(n_jobs=None)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def run_ridge(X_train, y_train, X_eval) -> np.ndarray:
    alphas = np.logspace(-3, 3, 13)
    # sklearn versions differ on RidgeCV(store_cv_values). Be version-safe.
    try:
        base = RidgeCV(alphas=alphas, store_cv_values=False)
    except TypeError:
        base = RidgeCV(alphas=alphas)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def run_nnls_coeff(X_train, y_train, X_eval) -> np.ndarray:
    base = LinearRegression(positive=True)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def run_mean(y_train, n_samples: int) -> np.ndarray:
    mu = y_train.mean(axis=0, keepdims=True)
    return np.repeat(mu, n_samples, axis=0)


def learn_signature_from_train(Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    # Solve min_S ||Y S - X||_F via least squares; clip and row-normalize.
    S, *_ = np.linalg.lstsq(ytr, Xtr, rcond=None)  # (K x G)
    S = np.clip(S, 0.0, None)
    rs = S.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return S / rs


def load_signature_csv(path: Path, expected_genes: List[str] | None) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    genes = list(df.columns)
    if expected_genes is not None and list(expected_genes) != genes:
        raise ValueError("Signature genes must match prepared gene order (selected_genes.txt)")
    S = df.values.astype(np.float64)
    S = np.clip(S, 0.0, None)
    rs = S.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (S / rs), list(df.index)


def _nnls_batch(S: np.ndarray, X: np.ndarray) -> np.ndarray:
    ST = S.T  # (G x K)
    N = X.shape[0]
    K = S.shape[0]
    W = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        w, _ = nnls(ST, X[i].astype(np.float64, copy=False))
        W[i] = w
    return W


def run_cibersort_like(Xtr, ytr, Xev, signature_path: str | None, genes: List[str] | None) -> np.ndarray:
    if signature_path:
        S, _ = load_signature_csv(Path(signature_path), genes)
    else:
        S = learn_signature_from_train(Xtr, ytr)
    return project_to_simplex(_nnls_batch(S, Xev))


def run_music_like(Xtr, ytr, Xev) -> np.ndarray:
    # Weighted NNLS using gene inverse-variance (bulk-only MuSiC approx).
    S = learn_signature_from_train(Xtr, ytr)
    gvar = np.var(Xtr, axis=0) + 1e-8
    w = 1.0 / np.sqrt(gvar)
    Sw = S * w
    Xw = Xev * w
    return project_to_simplex(_nnls_batch(Sw, Xw))


def run_bayesprism_like(Xtr, ytr, Xev, lam: float = 1e-2, iters: int = 200, step: float = 1e-1) -> np.ndarray:
    # Entropy-regularized deconvolution via projected gradient on simplex.
    S = learn_signature_from_train(Xtr, ytr)
    u = np.clip(ytr.mean(axis=0), 1e-6, None)
    u = u / u.sum()
    ST = S.T
    W = []
    for x in Xev:
        w = u.copy()
        for _ in range(iters):
            r = x - w @ S
            grad = -2.0 * (r @ ST) + lam * (np.log(np.clip(w, 1e-12, None)) - np.log(u))
            w = np.clip(w - step * grad, 0.0, None)
            s = w.sum()
            w = w / (s if s > 1e-12 else 1.0)
        W.append(w)
    return np.asarray(W, dtype=np.float64)


def run_scaden_like(Xtr, ytr, Xev, hidden=(512, 256), seed: int = 42, max_iter: int = 80) -> np.ndarray:
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                       max_iter=max_iter, random_state=seed, verbose=False)
    mlp.fit(Xtr, ytr)
    return mlp.predict(Xev)


def run_dtd_like(Xtr, ytr, Xev) -> np.ndarray:
    S = learn_signature_from_train(Xtr, ytr)
    return project_to_simplex(_nnls_batch(S, Xev))


def run_adtd_like(Xtr, ytr, Xev, alpha: float = 1.0) -> np.ndarray:
    # Ridge-refine signature towards initial S0: (Y^T Y + alpha I) S = Y^T X + alpha S0
    S0 = learn_signature_from_train(Xtr, ytr)
    YtY = ytr.T @ ytr
    K = YtY.shape[0]
    S = np.linalg.solve(YtY + alpha * np.eye(K), ytr.T @ Xtr + alpha * S0)
    S = np.clip(S, 0.0, None)
    rs = S.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    S = S / rs
    return project_to_simplex(_nnls_batch(S, Xev))


RUNNERS = {
    "ols": run_ols,
    "ridge": run_ridge,
    "nnls": run_nnls_coeff,
    "mean": lambda Xtr, ytr, Xev: run_mean(ytr, Xev.shape[0]),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Extended baseline deconvolution benchmarking")
    ap.add_argument("--data-dir", default="DECONOMIX_MODELS/Data/prepared", help="Folder with prepared arrays")
    ap.add_argument("--out", default="DECONOMIX_MODELS/results/Baselines_run", help="Output directory for results")
    ap.add_argument("--models", nargs="+", default=[
        "ols", "ridge", "nnls", "mean",
        "cibersort_like", "music_like", "bayesprism_like",
        "scaden_like", "dtd_like", "adtd_like"
    ])
    ap.add_argument("--split", choices=["test", "val"], default="test", help="Which split to evaluate")
    ap.add_argument("--signature", default=None, help="CSV signature (cell_types x genes) for cibersort_like")
    ap.add_argument("--bayes-lam", type=float, default=1e-2)
    ap.add_argument("--bayes-iters", type=int, default=200)
    ap.add_argument("--bayes-step", type=float, default=1e-1)
    ap.add_argument("--scaden-max-iter", type=int, default=80)
    ap.add_argument("--scaden-hidden", type=str, default="512,256")
    ap.add_argument("--adtd-alpha", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xv, yv, Xte, yte, cell_types = load_prepared(data_dir)
    genes = maybe_read_genes(data_dir)
    Xev, yev = (Xte, yte) if args.split == "test" else (Xv, yv)

    summary_rows: List[Dict[str, float]] = []

    print("=" * 70)
    print("BASELINE BENCHMARKS")
    print("=" * 70)
    print(f"Data: {data_dir}")
    print(f"Eval split: {args.split} -> {Xev.shape}")
    print(f"Cell types: {len(cell_types)}")
    print("Models:", ", ".join(args.models))

    for name in args.models:
        print(f"\n[Run] {name.upper()} ...")
        if name == "cibersort_like":
            yhat = run_cibersort_like(Xtr, ytr, Xev, signature_path=args.signature, genes=genes)
        elif name == "music_like":
            yhat = run_music_like(Xtr, ytr, Xev)
        elif name == "bayesprism_like":
            yhat = run_bayesprism_like(Xtr, ytr, Xev, lam=args.bayes_lam, iters=args.bayes_iters, step=args.bayes_step)
        elif name == "scaden_like":
            hidden = tuple(int(h) for h in args.scaden_hidden.split(",") if h)
            yhat = run_scaden_like(Xtr, ytr, Xev, hidden=hidden, max_iter=args.scaden_max_iter)
        elif name == "dtd_like":
            yhat = run_dtd_like(Xtr, ytr, Xev)
        elif name == "adtd_like":
            yhat = run_adtd_like(Xtr, ytr, Xev, alpha=args.adtd_alpha)
        else:
            runner = RUNNERS[name]
            yhat = runner(Xtr, ytr, Xev)

        df = evaluate(yev, yhat, cell_types)
        per_csv = out_root / f"performance_{name}.csv"
        df.to_csv(per_csv, index=False)
        avg = df.loc[df["cell_type"] == "AVERAGE"].iloc[0]
        print(f"  -> Avg Spearman={avg['spearman']:.3f}  MAE={avg['mae']:.4f}  saved={per_csv}")
        summary_rows.append({"model": name, "spearman": float(avg["spearman"]), "mae": float(avg["mae"])})

    summary = pd.DataFrame(summary_rows).sort_values(by="spearman", ascending=False)
    summary_csv = out_root / "summary_baselines.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n[Summary] {summary_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
