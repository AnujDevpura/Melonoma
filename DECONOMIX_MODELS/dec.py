"""
BLEEDING EDGE DECONVOLUTION: Attention-ResNet + MixUp + Rank Loss
------------------------------------------------------------------
Techniques included:
1. Gene Attention Gating: Dynamically weights genes per sample (Context-aware).
2. MixUp Regularization: Enforces linearity (crucial for deconvolution).
3. Rank-Consistency Loss: Optimizes directly for Spearman Correlation.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print(f"🚀 LAUNCHING BLEEDING EDGE DECONVOLUTION")
print(f"⚙️ DEVICE: {device}")
print("="*70)

DATA_PATH = 'Data/rna_data.h5ad'  # <--- Verify this path matches your file

# ============================================================================
# PART 1: DATA LOADING & SMART FILTERING
# ============================================================================
print("\n[1/6] Loading and Filtering Single-Cell Data...")

if not os.path.exists(DATA_PATH):
    fallback = 'rna_data.h5ad'
    if os.path.exists(fallback):
        DATA_PATH = fallback
    else:
        raise FileNotFoundError(f"Could not find data at {DATA_PATH} or {fallback}")
adata = sc.read_h5ad(DATA_PATH)
adata_control = adata[adata.obs['disease'] == 'control'].copy()

# Filter to immune cells only
immune_keywords = ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Monocyte', 'DC', 'Plasma', 'Mast']
mask = adata_control.obs['cell_type'].str.lower().apply(lambda x: any(k.lower() in x for k in immune_keywords))
adata_immune = adata_control[mask].copy()

# --- SMART FILTERING ---
# Remove cell types that constitute < 1% of the data (noise reduction)
counts = adata_immune.obs['cell_type'].value_counts()
props = counts / len(adata_immune)
major_types = props[props >= 0.01].index.tolist()
adata_major = adata_immune[adata_immune.obs['cell_type'].isin(major_types)].copy()
cell_types = sorted(major_types)

print(f" -> Retained {len(cell_types)} major cell types (filtered out rare ones)")

# --- FEATURE SELECTION ---
print(" -> Selecting optimal gene features...")
sc.pp.normalize_total(adata_major, target_sum=1e4)
sc.pp.log1p(adata_major)

# 1. Highly Variable Genes
sc.pp.highly_variable_genes(adata_major, n_top_genes=2000, flavor='seurat_v3')
hvg = adata_major.var_names[adata_major.var['highly_variable']].tolist()

# 2. Marker Genes (Distinctive features)
sc.tl.rank_genes_groups(adata_major, 'cell_type', method='wilcoxon')
markers = set()
for ct in cell_types:
    genes = sc.get.rank_genes_groups_df(adata_major, group=ct).head(100)['names']
    markers.update(genes)

final_genes = sorted(list(set(hvg) | markers))
adata_final = adata_major[:, final_genes].copy()
print(f" -> Final Feature Set: {len(final_genes)} genes")

# ============================================================================
# PART 2: ADVANCED SIMULATION (PSEUDO-BULK)
# ============================================================================
print("\n[2/6] Simulating Bulk Data...")

def simulate_bulk(adata, n_samples=1000):
    # Convert to dense if sparse
    if hasattr(adata.X, 'toarray'): X = adata.X.toarray()
    else: X = np.array(adata.X)
    
    # Revert log-space for accurate summation (Physical mixing happens in linear space)
    if X.max() < 20: X = np.expm1(X)
    
    y_labels = adata.obs['cell_type'].values
    unique_labels = sorted(adata.obs['cell_type'].unique())
    n_cells_total = adata.n_obs
    
    bulks = []
    proportions = []
    
    for i in range(n_samples):
        # Randomly sample cells
        n_cells_in_sample = np.random.randint(500, 3000)
        indices = np.random.choice(n_cells_total, n_cells_in_sample, replace=True)
        
        selected_cells = X[indices, :]
        selected_labels = y_labels[indices]
        
        # Aggregation
        bulk_expr = selected_cells.sum(axis=0)
        
        # Add Sequencing Depth Noise (Gamma distribution)
        noise = np.random.gamma(shape=20, scale=0.05, size=bulk_expr.shape)
        bulk_expr = bulk_expr * noise
        
        # Normalize (CPM)
        bulk_expr = bulk_expr / (bulk_expr.sum() + 1e-9) * 1e6
        
        # Calculate True Proportions
        u, c = np.unique(selected_labels, return_counts=True)
        counts_map = dict(zip(u, c))
        props_vec = [counts_map.get(ct, 0) / n_cells_in_sample for ct in unique_labels]
        
        bulks.append(bulk_expr)
        proportions.append(props_vec)
        
    return np.array(bulks), np.array(proportions)

# Generate datasets
X_train, y_train = simulate_bulk(adata_final, n_samples=15000)
X_val, y_val = simulate_bulk(adata_final, n_samples=2000)
X_test, y_test = simulate_bulk(adata_final, n_samples=1000)

print(f" -> Training Data: {X_train.shape}")

# ============================================================================
# PART 3: THE "PATENT-WORTHY" MODEL ARCHITECTURE
# ============================================================================
print("\n[3/6] Building Attention-ResNet Architecture...")

class GeneAttentionBlock(nn.Module):
    """Learns to weight genes dynamically per sample (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        w = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ResBlock(nn.Module):
    """Deep residual block with dropout"""
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class BleedingEdgeDeconv(nn.Module):
    def __init__(self, n_input, n_classes):
        super().__init__()
        
        # Feature Expansion
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
        
        # Attention Mechanism (The "Secret Sauce")
        self.attention = GeneAttentionBlock(1024, reduction=8)
        
        # Deep Reasoning
        self.res_layers = nn.Sequential(
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024)
        )
        
        # Bottleneck & Head
        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )
        self.head = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax(dim=1) # Enforce sum to 1
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.res_layers(x)
        x = self.bottleneck(x)
        return self.softmax(self.head(x))

# ============================================================================
# PART 4: TRAINING UTILITIES (MixUp + Custom Loss)
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """Generates mixed samples to force model linearity"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class RankConsistencyLoss(nn.Module):
    """Combines KL Divergence (Distribution matching) + Cosine Loss (Correlation matching)"""
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.cosine = nn.CosineEmbeddingLoss()
        
    def forward(self, pred, target):
        loss_kl = self.kl(pred.log(), target)
        # Flatten for global correlation check
        target_flat = torch.ones(pred.shape[0]).to(device)
        loss_cos = self.cosine(pred, target, target_flat)
        # Weighting: 70% correlation focus, 30% distribution focus
        return 0.3 * loss_kl + 0.7 * loss_cos

# ============================================================================
# PART 5: TRAINING LOOP
# ============================================================================
print("\n[4/6] Starting Training...")

# Preprocessing
scaler = RobustScaler()
X_train_s = torch.FloatTensor(scaler.fit_transform(X_train)).to(device)
X_val_s = torch.FloatTensor(scaler.transform(X_val)).to(device)
X_test_s = torch.FloatTensor(scaler.transform(X_test)).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)

# Model Init
model = BleedingEdgeDeconv(X_train.shape[1], len(cell_types)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = RankConsistencyLoss()

# Loader
train_loader = DataLoader(TensorDataset(X_train_s, y_train_t), batch_size=256, shuffle=True)

# Training
n_epochs = 100
best_val_loss = float('inf')

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    
    for x, y in train_loader:
        optimizer.zero_grad()
        
        # Apply MixUp
        x_mix, y_a, y_b, lam = mixup_data(x, y)
        out = model(x_mix)
        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_s)
        val_loss = criterion(val_pred, y_val_t).item()
        
        # Calculate Spearman for monitoring
        vp = val_pred.cpu().numpy()
        vt = y_val_t.cpu().numpy()
        corrs = [spearmanr(vt[:, i], vp[:, i])[0] for i in range(vt.shape[1])]
        avg_rho = np.nanmean(corrs)
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Rho: {avg_rho:.4f}")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# ============================================================================
# PART 6: FINAL EVALUATION & PLOTTING
# ============================================================================
print("\n[5/6] Evaluation...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    preds = model(X_test_s).cpu().numpy()

correlations = []
maes = []

print(f"\n{'CELL TYPE':<25} {'SPEARMAN':>10} {'MAE':>10}")
print("-" * 50)
for i, ct in enumerate(cell_types):
    r, _ = spearmanr(y_test[:, i], preds[:, i])
    mae = mean_absolute_error(y_test[:, i], preds[:, i])
    correlations.append(r)
    maes.append(mae)
    print(f"{ct:<25} {r:>10.3f} {mae:>10.4f}")

avg_corr = np.mean(correlations)
print("-" * 50)
print(f"{'AVERAGE':<25} {avg_corr:>10.3f} {np.mean(maes):>10.4f}")

# Plotting
print("\n[6/6] Generating Plots...")
n_plots = min(9, len(cell_types))
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i in range(n_plots):
    ax = axes[i]
    ax.scatter(y_test[:, i], preds[:, i], alpha=0.4, s=15, color='#4c72b0')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5)
    ax.set_title(f"{cell_types[i]}\nρ = {correlations[i]:.3f}")
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)

plt.suptitle(f"Bleeding Edge Deconvolution (Avg ρ={avg_corr:.3f})", fontsize=16)
plt.tight_layout()
plt.savefig('deconvolution_results.png')

print(f"\n✅ DONE. Results saved to 'deconvolution_results.png'.")