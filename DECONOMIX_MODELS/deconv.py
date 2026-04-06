import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("DEEP LEARNING DECONVOLUTION - SMART FILTERING")
print("="*70)

print("\n[STEP 1] Loading skin atlas data...")
p = os.path.join('Data','rna_data.h5ad')
adata = sc.read_h5ad(p if os.path.exists(p) else 'rna_data.h5ad')
adata_control = adata[adata.obs['disease'] == 'control'].copy()

immune_keywords = ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Monocyte',
                   'DC', 'Plasma cell', 'Mast cell']
mask = adata_control.obs['cell_type'].str.lower().apply(
    lambda x: any(k.lower() in x for k in immune_keywords)
)
adata_immune = adata_control[mask].copy()

print(f"Immune cells: {adata_immune.n_obs:,}")

print("\n[STEP 2] Filtering to abundant cell types...")

cell_type_counts = adata_immune.obs['cell_type'].value_counts()
cell_type_props = cell_type_counts / len(adata_immune)

print("\nCell type proportions:")
for ct, prop in cell_type_props.items():
    print(f"  {ct}: {prop*100:.2f}% ({cell_type_counts[ct]:,} cells)")

min_proportion = 0.01
major_cell_types = cell_type_props[cell_type_props >= min_proportion].index.tolist()

print(f"\nKeeping {len(major_cell_types)} major cell types (>{min_proportion*100}%):")
for ct in major_cell_types:
    print(f"  {ct}")

adata_major = adata_immune[adata_immune.obs['cell_type'].isin(major_cell_types)].copy()
print(f"\nFiltered to {adata_major.n_obs:,} cells in {len(major_cell_types)} cell types")

print("\n[STEP 3] Selecting marker genes...")

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

print("\n[STEP 4] Creating reference...")

if hasattr(adata_filtered.X, 'toarray'):
    X_data = adata_filtered.X.toarray()
else:
    X_data = np.array(adata_filtered.X)

is_logged = X_data.max() < 20
if is_logged:
    X_data = np.expm1(X_data)

reference_list = []
for ct in cell_types:
    mask = (adata_filtered.obs['cell_type'] == ct).values
    ct_data = X_data[mask, :]

    cell_sums = ct_data.sum(axis=1, keepdims=True)
    ct_data_cpm = ct_data / (cell_sums + 1e-9) * 1e6

    mean_expr = ct_data_cpm.mean(axis=0)
    reference_list.append(mean_expr)

reference_matrix = np.array(reference_list).T
print(f"Reference: {reference_matrix.shape}")

print("\n[STEP 5] Simulating pseudo-bulk samples...")

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

X_train_bulk, y_train_props = simulate_bulk_advanced(adata_filtered, n_samples=15000, seed=42)
X_val_bulk, y_val_props = simulate_bulk_advanced(adata_filtered, n_samples=2000, seed=123)
X_test_bulk, y_test_props = simulate_bulk_advanced(adata_filtered, n_samples=1000, seed=456)

print(f"\nTraining: {X_train_bulk.shape}")
print(f"Validation: {X_val_bulk.shape}")
print(f"Test: {X_test_bulk.shape}")

print("\n[STEP 6] Building neural network...")

class DeepDeconvolutionNet(nn.Module):
    def __init__(self, n_genes, n_cell_types):
        super(DeepDeconvolutionNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(n_genes, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_cell_types),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

n_genes = X_train_bulk.shape[1]
n_cell_types = len(cell_types)

model = DeepDeconvolutionNet(n_genes, n_cell_types)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n[STEP 7] Training...")

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
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

n_epochs = 200
best_val_loss = float('inf')
patience_counter = 0
max_patience = 30

print("\nTraining progress:")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Corr':>12}")
print("-" * 48)

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).numpy()
        val_loss = criterion(model(X_val_t), y_val_t).item()

        val_corrs = []
        for i in range(n_cell_types):
            corr, _ = spearmanr(y_val_props[:, i], val_preds[:, i])
            val_corrs.append(corr)
        avg_val_corr = np.mean(val_corrs)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_loss:12.6f} {avg_val_corr:12.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_filtered.pth')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load('best_model_filtered.pth'))

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t).numpy()

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
    ax.scatter(y_test_props[:, i], test_predictions[:, i], alpha=0.5, s=30, c='steelblue')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
    
    ax.set_xlabel('True Proportion', fontsize=10)
    ax.set_ylabel('Predicted Proportion', fontsize=10)
    ax.set_title(f'{cell_types[i]}\nρ = {correlations[i]:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)

for idx in range(len(cell_types), 9):
    axes[idx].axis('off')

plt.suptitle(f'Deep Learning (Filtered) - Avg ρ = {avg_corr:.3f}',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('deep_learning_filtered.png', dpi=300, bbox_inches='tight')
print("Saved: deep_learning_filtered.png")

perf_df = pd.DataFrame({
    'cell_type': cell_types,
    'spearman_correlation': correlations,
    'mae': mae_scores,
    'mean_proportion': y_test_props.mean(axis=0)
})
perf_df.to_csv('performance_filtered.csv', index=False)

import pickle
with open('scaler_filtered.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('cell_types_filtered.txt', 'w') as f:
    for ct in cell_types:
        f.write(f"{ct}\n")

np.save('reference_filtered.npy', reference_matrix)

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"\nFINAL RESULTS:")
print(f"  Average Spearman: {avg_corr:.3f}")
print(f"  Cell types: {len(cell_types)}")
print(f"  Training samples: 15,000")

print("\n" + "="*70)