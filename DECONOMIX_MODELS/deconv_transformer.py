import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy
import random, json, pickle
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
warnings.filterwarnings('ignore')
use_precomputed = False

# Optional JSON config support (overrides env defaults when provided)
parser = argparse.ArgumentParser(description='Transformer deconvolution')
parser.add_argument('--config', help='Path to JSON config for data/run/train/model settings')
known, _ = parser.parse_known_args()
CFG = None
if known.config:
    import json
    with open(known.config, 'r', encoding='utf-8-sig') as f:
        CFG = json.load(f)
else:
    # Load default config next to this script if present
    default_cfg = os.path.join(os.path.dirname(__file__), 'configs', 'transformer.json')
    if os.path.exists(default_cfg):
        import json
        with open(default_cfg, 'r', encoding='utf-8-sig') as f:
            CFG = json.load(f)
        print(f"[Config] Using default config: {default_cfg}")

def cfg_get(path, env_key=None, default=None):
    # path like ('data','prepared')
    if CFG is not None:
        d = CFG
        for k in path:
            if not isinstance(d, dict) or k not in d:
                break
            d = d[k]
        else:
            return d
    if env_key is not None:
        return os.getenv(env_key, default)
    return default

prepared_dir = cfg_get(('data','prepared'), 'DECONOMIX_PREPARED', None)
dataset_path = cfg_get(('data','h5ad'), 'DECONOMIX_DATA', 'Data/rna_data.h5ad')
outdir = cfg_get(('run','outdir'), 'DECONOMIX_OUTDIR', '.')
os.makedirs(outdir, exist_ok=True)
progress_enabled = bool(cfg_get(('run','progress'), 'DECONOMIX_PROGRESS', '1') != '0')
seed_env = cfg_get(('run','seed'), 'DECONOMIX_SEED', None)

# GPU / performance setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# Optional reproducibility via env DECONOMIX_SEED
seed_env = cfg_get(('run','seed'), 'DECONOMIX_SEED', None)
if seed_env is not None:
    try:
        seed = int(seed_env)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
    except Exception:
        pass

print("="*70)
print("TRANSFORMER DECONVOLUTION")
print("="*70)

print("\n[STEP 1] Loading skin atlas data or precomputed arrays...")
if prepared_dir and os.path.exists(os.path.join(prepared_dir, 'X_train.npy')):
    use_precomputed = True
    print(f"Using precomputed arrays from: {prepared_dir}")
    X_train_bulk = np.load(os.path.join(prepared_dir, 'X_train.npy'))
    y_train_props = np.load(os.path.join(prepared_dir, 'y_train.npy'))
    X_val_bulk = np.load(os.path.join(prepared_dir, 'X_val.npy'))
    y_val_props = np.load(os.path.join(prepared_dir, 'y_val.npy'))
    X_test_bulk = np.load(os.path.join(prepared_dir, 'X_test.npy'))
    y_test_props = np.load(os.path.join(prepared_dir, 'y_test.npy'))
    with open(os.path.join(prepared_dir, 'cell_types.txt'), 'r', encoding='utf-8') as f:
        cell_types = [line.strip() for line in f if line.strip()]
else:
    adata = sc.read_h5ad(dataset_path)
if not use_precomputed:
    adata_control = adata[adata.obs['disease'] == 'control'].copy()

    immune_keywords = ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Monocyte',
                       'DC', 'Plasma cell', 'Mast cell']
    mask = adata_control.obs['cell_type'].str.lower().apply(
        lambda x: any(k.lower() in x for k in immune_keywords)
    )
    adata_immune = adata_control[mask].copy()

    print(f"Immune cells: {adata_immune.n_obs:,}")

if not use_precomputed:
    print("\n[STEP 2] Filtering to abundant cell types...")

if not use_precomputed:
    cell_type_counts = adata_immune.obs['cell_type'].value_counts()
    cell_type_props = cell_type_counts / len(adata_immune)

    min_proportion = 0.01
    major_cell_types = cell_type_props[cell_type_props >= min_proportion].index.tolist()

    print(f"\nKeeping {len(major_cell_types)} major cell types (>{min_proportion*100}%):")
    for ct in major_cell_types:
        print(f"  {ct}")

    adata_major = adata_immune[adata_immune.obs['cell_type'].isin(major_cell_types)].copy()
    print(f"\nFiltered to {adata_major.n_obs:,} cells in {len(major_cell_types)} cell types")

if not use_precomputed:
    print("\n[STEP 3] Selecting marker genes...")

if not use_precomputed:
    adata_norm = adata_major.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    sc.pp.highly_variable_genes(adata_norm, n_top_genes=1500, flavor='seurat_v3')
    hvg = adata_norm.var_names[adata_norm.var['highly_variable']].tolist()

    sc.tl.rank_genes_groups(adata_norm, 'cell_type', method='wilcoxon')

    marker_genes = set()
    cell_types = sorted(major_cell_types)

    for ct in cell_types:
        genes = sc.get.rank_genes_groups_df(adata_norm, group=ct).head(150)['names']
        marker_genes.update(genes)

    selected_genes = sorted(list(set(hvg) | marker_genes))
    print(f"Selected genes: {len(selected_genes)}")

    adata_filtered = adata_major[:, selected_genes].copy()

# ============================================================================
# STEP 4: SIMULATE TRAINING DATA
# ============================================================================
if not use_precomputed:
    print("\n[STEP 4] Simulating pseudo-bulk samples...")

def simulate_bulk_advanced(adata, n_samples=10000, seed=42):
    np.random.seed(seed)

    if hasattr(adata.X, 'toarray'):
        X_data = adata.X.toarray()
    else:
        X_data = np.array(adata.X)

    is_logged = X_data.max() < 20
    if is_logged:
        X_data = np.expm1(X_data)

    cell_type_array = adata.obs['cell_type'].values
    cell_types_list = sorted(adata.obs['cell_type'].unique())

    bulks = []
    props = []

    print(f"Generating {n_samples} samples...")
    for i in range(n_samples):
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_samples}...")

        n_cells = np.random.randint(500, 3000)
        indices = np.random.choice(adata.n_obs, n_cells, replace=True)
        cells = X_data[indices, :]

        bulk = cells.sum(axis=0)

        noise_level = np.random.uniform(0.01, 0.05)
        bulk = bulk * (1 + np.random.normal(0, noise_level, bulk.shape))
        bulk = np.maximum(bulk, 0)

        bulk = bulk / (bulk.sum() + 1e-9) * 1e6

        bulks.append(bulk)

        sampled_cts = cell_type_array[indices]
        unique, counts = np.unique(sampled_cts, return_counts=True)
        prop_dict = dict(zip(unique, counts / n_cells))
        prop_vec = [prop_dict.get(ct, 0.0) for ct in cell_types_list]
        props.append(prop_vec)

    return np.array(bulks), np.array(props)

if not use_precomputed:
    X_train_bulk, y_train_props = simulate_bulk_advanced(adata_filtered, n_samples=15000, seed=42)
    X_val_bulk, y_val_props = simulate_bulk_advanced(adata_filtered, n_samples=2000, seed=123)
    X_test_bulk, y_test_props = simulate_bulk_advanced(adata_filtered, n_samples=1000, seed=456)

print(f"\nTraining: {X_train_bulk.shape}")
print(f"Validation: {X_val_bulk.shape}")
print(f"Test: {X_test_bulk.shape}")

print("\n[STEP 5] Building Transformer model...")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class TransformerDeconvolution(nn.Module):
    def __init__(self, n_genes, n_cell_types, d_model=512, num_heads=8, num_layers=4, chunk_size=128):
        super(TransformerDeconvolution, self).__init__()

        self.chunk_size = chunk_size
        self.n_chunks = (n_genes + chunk_size - 1) // chunk_size

        self.gene_embedding = nn.Linear(chunk_size, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_chunks, d_model))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_model*4, dropout=0.3)
            for _ in range(num_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        use_dirichlet = bool(cfg_get(('train','use_dirichlet'), None, False))
        head_layers = [
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, n_cell_types),
        ]
        # Softmax for simplex if not using Dirichlet; Dirichlet path will map to alpha via softplus in loss
        if not use_dirichlet:
            head_layers.append(nn.Softmax(dim=1))
        self.decoder = nn.Sequential(*head_layers)

    def forward(self, x):
        batch_size = x.size(0)

        n_genes = x.size(1)
        pad_size = (self.chunk_size - n_genes % self.chunk_size) % self.chunk_size
        if pad_size > 0:
            x = torch.cat([x, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)

        x = x.view(batch_size, self.n_chunks, self.chunk_size)

        x = self.gene_embedding(x)

        x = x + self.pos_embedding

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)

        output = self.decoder(x)

        return output

n_genes = X_train_bulk.shape[1]
n_cell_types = len(cell_types)

# Model dims via config when provided
d_model = int(cfg_get(('model','d_model'), None, 512))
num_heads = int(cfg_get(('model','num_heads'), None, 8))
num_layers = int(cfg_get(('model','num_layers'), None, 4))
model = TransformerDeconvolution(n_genes, n_cell_types, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
model.to(device)

# EMA model for smoother validation and checkpoints
ema_decay = float(cfg_get(('train','ema_decay'), 'DECONOMIX_EMA_DECAY', '0.999'))
ema_model = deepcopy(model).to(device)
for p in ema_model.parameters():
    p.requires_grad_(False)

@torch.no_grad()
def ema_update(ema_m, model_m, decay: float):
    for p_ema, p in zip(ema_m.parameters(), model_m.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

print("\n[STEP 6] Training...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_bulk)
X_val_scaled = scaler.transform(X_val_bulk)
X_test_scaled = scaler.transform(X_test_bulk)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_props)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val_props)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_props)

train_dataset = TensorDataset(X_train_t, y_train_t)
batch_size = int(cfg_get(('train','batch_size'), 'DECONOMIX_BATCH', '512'))
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == 'cuda'),
)

eps = 1e-6
avg_props = y_train_props.mean(axis=0)
class_weights_np = 1.0 / (avg_props + eps)
class_weights_np = class_weights_np / class_weights_np.mean()
cap = float(cfg_get(('train','weight_cap'), 'DECONOMIX_WEIGHT_CAP', '3.0'))
if cap > 0:
    class_weights_np = np.minimum(class_weights_np, cap)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

label_smooth = float(cfg_get(('train','label_smooth'), 'DECONOMIX_LABEL_SMOOTH', '0.01'))

def apply_label_smoothing(y: torch.Tensor, smooth: float) -> torch.Tensor:
    if smooth <= 0:
        return y
    c = y.size(1)
    return (1.0 - smooth) * y + smooth / c

def weighted_mse(pred, target):
    loss = (pred - target) ** 2
    loss = loss * class_weights
    return loss.mean()

max_lr = float(cfg_get(('train','lr_max'), None, 0.001))
weight_decay = float(cfg_get(('train','weight_decay'), None, 1e-4))
optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

n_epochs = int(cfg_get(('train','epochs'), None, 200))
best_val_corr = -1.0
patience_counter = 0
max_patience = int(cfg_get(('train','patience'), None, 60))
scaler_amp = GradScaler(enabled=(device.type == 'cuda'))
scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=n_epochs, steps_per_epoch=len(train_loader))
mixup_alpha = float(cfg_get(('train','mixup_alpha'), 'DECONOMIX_MIXUP_ALPHA', '0.0'))
use_mixup = mixup_alpha > 0
grad_clip = float(cfg_get(('train','gradient_clip'), None, 0.0))
use_swa = bool(cfg_get(('train','use_swa'), None, False))
swa_snapshots = int(cfg_get(('train','swa_snapshots'), None, 3))
cos_w = float(cfg_get(('train','cosine_weight'), None, 0.0))
corr_w = float(cfg_get(('train','corr_weight'), None, 0.0))
if use_swa:
    swa_model = AveragedModel(ema_model)
    swa_count = 0

print("\nTraining progress:")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Corr':>12}")
print("-" * 48)

epoch_iter = range(n_epochs)
if progress_enabled:
    epoch_iter = tqdm(epoch_iter, desc="Epochs", dynamic_ncols=True)

for epoch in epoch_iter:
    model.train()
    train_loss = 0
    batch_iter = train_loader
    if progress_enabled:
        batch_iter = tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False, dynamic_ncols=True)

    for batch_X, batch_y in batch_iter:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Optional MixUp
        if use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(batch_X.size(0), device=device)
            batch_X_m = lam * batch_X + (1.0 - lam) * batch_X[idx]
            batch_y_m = lam * batch_y + (1.0 - lam) * batch_y[idx]
        else:
            batch_X_m, batch_y_m = batch_X, batch_y
        # Label smoothing
        batch_y_m = apply_label_smoothing(batch_y_m, label_smooth)
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(batch_X_m)
            # Choose loss: Dirichlet NLL or weighted MSE
            if bool(cfg_get(('train','use_dirichlet'), None, False)):
                # Convert head outputs to Dirichlet concentration alpha via softplus
                dirichlet_eps = float(cfg_get(('train','dirichlet_eps'), None, 1e-6))
                label_tau = float(cfg_get(('train','label_tau'), None, 1e-4))
                alpha = F.softplus(outputs) + dirichlet_eps
                y_s = apply_label_smoothing(batch_y_m, label_tau)
                y_s = torch.clamp(y_s, min=dirichlet_eps)
                # Dirichlet NLL
                loss = (
                    torch.lgamma(alpha.sum(dim=1))
                    - torch.lgamma(alpha).sum(dim=1)
                    - ((alpha - 1.0) * torch.log(y_s)).sum(dim=1)
                ).mean()
            else:
                loss = weighted_mse(outputs, batch_y_m)
            # Cosine auxiliary loss
            if cos_w > 0:
                # For Dirichlet, compare normalized alpha means; else use outputs directly
                if bool(cfg_get(('train','use_dirichlet'), None, False)):
                    p = alpha / alpha.sum(dim=1, keepdim=True)
                else:
                    p = outputs
                cos_loss = (1 - F.cosine_similarity(p, batch_y_m, dim=1)).mean()
                loss = loss + cos_w * cos_loss
            # Correlation-aware loss (negative Pearson correlation)
            if corr_w > 0:
                if bool(cfg_get(('train','use_dirichlet'), None, False)):
                    p = alpha / alpha.sum(dim=1, keepdim=True)
                # else p already assigned above when cos_w>0; ensure p exists
                else:
                    p = outputs
                px = p - p.mean(dim=0, keepdim=True)
                yx = batch_y_m - batch_y_m.mean(dim=0, keepdim=True)
                cov = (px * yx).mean(dim=0)
                stdp = (px.pow(2).mean(dim=0) + 1e-8).sqrt()
                stdy = (yx.pow(2).mean(dim=0) + 1e-8).sqrt()
                corr = cov / (stdp * stdy)
                corr_loss = 1.0 - corr.mean()
                loss = loss + corr_w * corr_loss
        scaler_amp.scale(loss).backward()
        if grad_clip > 0:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()
        ema_update(ema_model, model, ema_decay)
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad(), autocast(enabled=(device.type == 'cuda')):
        # Validate with EMA (or SWA-on-EMA) weights
        eval_model = ema_model
        if use_swa and (epoch + 1) >= n_epochs - swa_snapshots:
            # accumulate SWA snapshots from EMA in the tail epochs
            swa_model.update_parameters(ema_model)
            swa_count += 1
            eval_model = swa_model
        val_out = eval_model(X_val_t.to(device))
        if bool(cfg_get(('train','use_dirichlet'), None, False)):
            alpha_v = F.softplus(val_out) + float(cfg_get(('train','dirichlet_eps'), None, 1e-6))
            p_v = alpha_v / alpha_v.sum(dim=1, keepdim=True)
            val_preds = p_v.detach().cpu().numpy()
            # compute proxy loss against smoothed labels
            yv_s = apply_label_smoothing(y_val_t.to(device), float(cfg_get(('train','label_tau'), None, 1e-4)))
            val_loss = (
                torch.lgamma(alpha_v.sum(dim=1))
                - torch.lgamma(alpha_v).sum(dim=1)
                - ((alpha_v - 1.0) * torch.log(torch.clamp(yv_s, min=1e-6))).sum(dim=1)
            ).mean().item()
        else:
            val_preds = val_out.detach().cpu().numpy()
            val_loss = weighted_mse(val_out, y_val_t.to(device)).item()

        val_corrs = []
        for i in range(n_cell_types):
            corr, _ = spearmanr(y_val_props[:, i], val_preds[:, i])
            val_corrs.append(corr)
        avg_val_corr = np.mean(val_corrs)

    # OneCycleLR is stepped per batch; no val step here

    if progress_enabled:
        try:
            epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", corr=f"{avg_val_corr:.3f}")
        except Exception:
            pass

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_loss:12.6f} {avg_val_corr:12.3f}")

    if avg_val_corr > best_val_corr + 1e-4:
        best_val_corr = avg_val_corr
        patience_counter = 0
        os.makedirs(outdir, exist_ok=True)
        # Save EMA (or SWA-EMA) weights as checkpoint
        to_save = ema_model
        if use_swa and swa_count > 0:
            to_save = swa_model
        torch.save(to_save.state_dict(), os.path.join(outdir, 'best_model_transformer.pth'))
        try:
            with open(os.path.join(outdir, 'transformer_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            cfg = {
                'batch_size': batch_size,
                'optimizer': 'AdamW',
                'lr_max': 0.001,
                'weight_decay': 1e-4,
                'epochs': n_epochs,
                'scheduler': 'OneCycleLR',
                'class_weights': class_weights_np.tolist(),
                'best_val_corr': float(best_val_corr),
                'ema_decay': ema_decay,
                'mixup_alpha': mixup_alpha,
                'label_smooth': label_smooth,
                'seed': seed_env,
            }
            with open(os.path.join(outdir, 'transformer_config.json'), 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

outdir = outdir
model.load_state_dict(torch.load(os.path.join(outdir, 'best_model_transformer.pth')))

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t.to(device)).detach().cpu().numpy()

print(f"\n{'Cell Type':<30} {'Spearman ρ':>12} {'MAE':>10} {'Avg Prop':>10}")
print("-" * 64)

correlations = []
mae_scores = []

for i, ct in enumerate(cell_types):
    corr, _ = spearmanr(y_test_props[:, i], test_predictions[:, i])
    correlations.append(corr)

    mae = mean_absolute_error(y_test_props[:, i], test_predictions[:, i])
    mae_scores.append(mae)

    avg_prop = y_test_props[:, i].mean()

    print(f"{ct:<30} {corr:>12.3f} {mae:>10.4f} {avg_prop:>10.4f}")

print("-" * 64)
avg_corr = np.mean(correlations)
print(f"{'AVERAGE':<30} {avg_corr:>12.3f} {np.mean(mae_scores):>10.4f}")

print("\n[Creating visualizations...]")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, i in enumerate(range(min(9, len(cell_types)))):
    ax = axes[idx]
    ax.scatter(y_test_props[:, i], test_predictions[:, i], alpha=0.5, s=30, c='coral')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)

    ax.set_xlabel('True Proportion', fontsize=10)
    ax.set_ylabel('Predicted Proportion', fontsize=10)
    ax.set_title(f'{cell_types[i]}\nρ = {correlations[i]:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)

for idx in range(len(cell_types), 9):
    axes[idx].axis('off')

plt.suptitle(f'Transformer Deconvolution - Avg ρ = {avg_corr:.3f}',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'transformer_deconvolution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(outdir, 'transformer_deconvolution.png')}")

perf_df = pd.DataFrame({
    'cell_type': cell_types,
    'spearman_correlation': correlations,
    'mae': mae_scores,
    'mean_proportion': y_test_props.mean(axis=0)
})
perf_df.to_csv(os.path.join(outdir, 'performance_transformer.csv'), index=False)

print("\n" + "="*70)
print("TRANSFORMER COMPLETE")
print("="*70)
print(f"\nFINAL RESULTS:")
print(f"  Average Spearman: {avg_corr:.3f}")
print(f"  Architecture: Multi-head attention with {sum(p.numel() for p in model.parameters()):,} parameters")

print("\n" + "="*70)
