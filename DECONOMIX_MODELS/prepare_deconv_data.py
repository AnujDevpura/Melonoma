"""
Prepare deconvolution data once so model scripts can start training immediately.

Outputs (default: Data/prepared):
  - X_train.npy, y_train.npy (n_samples x n_genes / n_cell_types)
  - X_val.npy,   y_val.npy
  - X_test.npy,  y_test.npy
  - cell_types.txt (one per line)
  - selected_genes.txt
  - adjacency_matrix.npy (unnormalized, for GNN)

Usage (from project root):
  python prepare_deconv_data.py --data Data/rna_data.h5ad --out Data/prepared \
    --n-train 15000 --n-val 2000 --n-test 1000 --hvg 1500 --markers 150

Model scripts will automatically use these files if DECONOMIX_PREPARED points
to the output directory (or if Data/prepared exists by default).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import scanpy as sc


IMMUNE_KEYWORDS = [
    't cell', 'b cell', 'nk cell', 'macrophage', 'monocyte',
    'dc', 'plasma cell', 'mast cell'
]


def filter_control_immune(adata: sc.AnnData) -> sc.AnnData:
    adata = adata[adata.obs['disease'].astype(str).str.lower() == 'control'].copy()
    mask = adata.obs['cell_type'].astype(str).str.lower().apply(
        lambda x: any(k in x for k in IMMUNE_KEYWORDS)
    )
    return adata[mask].copy()


def select_major_cell_types(adata: sc.AnnData, min_prop: float = 0.01) -> Tuple[sc.AnnData, List[str]]:
    counts = adata.obs['cell_type'].value_counts()
    props = counts / len(adata)
    major = props[props >= min_prop].index.tolist()
    return adata[adata.obs['cell_type'].isin(major)].copy(), sorted(major)


def choose_genes(adata: sc.AnnData, n_hvg: int = 1500, n_markers: int = 150, cell_types: List[str] = None) -> List[str]:
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor='seurat_v3')
    hvg = ad.var_names[ad.var['highly_variable']].tolist()

    sc.tl.rank_genes_groups(ad, 'cell_type', method='wilcoxon')
    markers = set()
    groups = cell_types or sorted(adata.obs['cell_type'].unique())
    for ct in groups:
        df = sc.get.rank_genes_groups_df(ad, group=ct).head(n_markers)
        markers.update(df['names'])

    genes = sorted(set(hvg) | markers)
    return genes


def simulate_bulk_advanced(adata: sc.AnnData, n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    is_logged = X.max() < 20
    if is_logged:
        X = np.expm1(X)

    cell_type_array = adata.obs['cell_type'].values
    cell_types_list = sorted(adata.obs['cell_type'].unique())

    bulks = []
    props = []
    for i in range(n_samples):
        n_cells = np.random.randint(500, 3000)
        idx = np.random.choice(adata.n_obs, n_cells, replace=True)
        cells = X[idx, :]
        bulk = cells.sum(axis=0)
        noise_level = np.random.uniform(0.01, 0.05)
        bulk = bulk * (1 + np.random.normal(0, noise_level, bulk.shape))
        bulk = np.maximum(bulk, 0)
        bulk = bulk / (bulk.sum() + 1e-9) * 1e6
        bulks.append(bulk)

        sampled_cts = cell_type_array[idx]
        u, c = np.unique(sampled_cts, return_counts=True)
        prop_dict = dict(zip(u, c / n_cells))
        props.append([prop_dict.get(ct, 0.0) for ct in cell_types_list])

    return np.asarray(bulks, dtype=np.float32), np.asarray(props, dtype=np.float32)


def build_adjacency(adata: sc.AnnData, selected_genes: List[str], threshold: float = 0.3) -> np.ndarray:
    if hasattr(adata[:, selected_genes].X, 'toarray'):
        expr = adata[:, selected_genes].X.toarray()
    else:
        expr = np.array(adata[:, selected_genes].X)

    n_sub = min(5000, expr.shape[0])
    idx = np.random.choice(expr.shape[0], n_sub, replace=False)
    expr_sub = expr[idx, :]

    corr = np.corrcoef(expr_sub.T)
    corr = np.nan_to_num(corr, 0.0)
    A = (np.abs(corr) > threshold).astype(float)
    A = A + np.eye(A.shape[0])
    return A.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare deconvolution data once')
    ap.add_argument('--data', default=os.getenv('DECONOMIX_DATA', 'Data/rna_data.h5ad'))
    ap.add_argument('--out', default='Data/prepared')
    ap.add_argument('--n-train', type=int, default=15000)
    ap.add_argument('--n-val', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--hvg', type=int, default=1500)
    ap.add_argument('--markers', type=int, default=150)
    ap.add_argument('--adj', action='store_true', default=True, help='Precompute adjacency for GNN')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print('=' * 70)
    print('PREPARING DECONVOLUTION DATA')
    print('=' * 70)

    print(f"[1] Reading: {args.data}")
    adata = sc.read_h5ad(args.data)
    print(f"    Loaded: cells={adata.n_obs:,}, genes={adata.n_vars:,}")

    print('[2] Filtering to control + immune...')
    adata = filter_control_immune(adata)
    adata, major_ct = select_major_cell_types(adata, min_prop=0.01)
    print(f"    After filter: cells={adata.n_obs:,}, types={len(major_ct)}")

    print('[3] Selecting genes (HVG + markers)...')
    genes = choose_genes(adata, n_hvg=args.hvg, n_markers=args.markers, cell_types=major_ct)
    adata = adata[:, genes].copy()
    print(f"    Genes selected: {len(genes)}")

    print('[4] Simulating pseudo-bulk (train/val/test)...')
    X_train, y_train = simulate_bulk_advanced(adata, n_samples=args.n_train, seed=42)
    X_val, y_val = simulate_bulk_advanced(adata, n_samples=args.n_val, seed=123)
    X_test, y_test = simulate_bulk_advanced(adata, n_samples=args.n_test, seed=456)
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print('[5] Saving arrays and metadata...')
    np.save(os.path.join(args.out, 'X_train.npy'), X_train)
    np.save(os.path.join(args.out, 'y_train.npy'), y_train)
    np.save(os.path.join(args.out, 'X_val.npy'), X_val)
    np.save(os.path.join(args.out, 'y_val.npy'), y_val)
    np.save(os.path.join(args.out, 'X_test.npy'), X_test)
    np.save(os.path.join(args.out, 'y_test.npy'), y_test)
    with open(os.path.join(args.out, 'cell_types.txt'), 'w', encoding='utf-8') as f:
        for ct in major_ct:
            f.write(f"{ct}\n")
    with open(os.path.join(args.out, 'selected_genes.txt'), 'w', encoding='utf-8') as f:
        for g in genes:
            f.write(f"{g}\n")

    if args.adj:
        print('[6] Precomputing adjacency for GNN...')
        A = build_adjacency(adata, genes, threshold=0.3)
        np.save(os.path.join(args.out, 'adjacency_matrix.npy'), A)
        print(f"    Adjacency: {A.shape}, nnz≈{int(A.sum())}")

    print('\nDone. Set environment and train:')
    print(f"  $env:DECONOMIX_PREPARED = '{os.path.abspath(args.out)}'")
    print(f"  $env:DECONOMIX_DATA = '{os.path.abspath(args.data)}'")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

