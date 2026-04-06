"""VAE deconvolution with disentangled latent representations."""

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
import torch.nn.functional as F
import warnings
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
warnings.filterwarnings('ignore')
use_precomputed = False
prepared_dir = os.getenv('DECONOMIX_PREPARED')
progress_enabled = os.getenv('DECONOMIX_PROGRESS', '1') != '0'

# GPU / performance setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

print("="*70)
print("VAE DECONVOLUTION - LATENT SPACE MODELING")
print("="*70)

print("\n[STEP 1] Loading skin atlas data or precomputed arrays...")
dataset_path = os.getenv('DECONOMIX_DATA', 'Data/rna_data.h5ad')
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

print("\n[STEP 5] Building VAE model...")

class VAEDeconvolution(nn.Module):
    def __init__(self, n_genes, n_cell_types, latent_dim=128):
        super(VAEDeconvolution, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(2048, n_genes)
        )

        self.proportion_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(128, n_cell_types),
            nn.Softmax(dim=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        proportions = self.proportion_predictor(z)
        return x_recon, proportions, mu, logvar

def vae_loss(x_recon, x, proportions, y_true, mu, logvar, beta=0.1, gamma=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    prop_loss = F.mse_loss(proportions, y_true, reduction='mean')
    total_loss = recon_loss + beta * kl_loss + gamma * prop_loss
    return total_loss, recon_loss, kl_loss, prop_loss

n_genes = X_train_bulk.shape[1]
n_cell_types = len(cell_types)

model = VAEDeconvolution(n_genes, n_cell_types, latent_dim=128)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
model.to(device)

print("\n[STEP 6] Training VAE...")

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
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    pin_memory=(device.type == 'cuda'),
)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

n_epochs = 200
best_val_loss = float('inf')
patience_counter = 0
max_patience = 30

print("\nTraining progress:")
print(f"{'Epoch':>6} {'Total Loss':>12} {'Recon':>10} {'KL':>10} {'Prop':>10} {'Val Corr':>12}")
print("-" * 70)

epoch_iter = range(n_epochs)
if progress_enabled:
    epoch_iter = tqdm(epoch_iter, desc="Epochs", dynamic_ncols=True)

for epoch in epoch_iter:
    model.train()
    total_loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    prop_loss_epoch = 0

    batch_iter = train_loader
    if progress_enabled:
        batch_iter = tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False, dynamic_ncols=True)

    for batch_X, batch_y in batch_iter:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()

        x_recon, proportions, mu, logvar = model(batch_X)
        loss, recon_loss, kl_loss, prop_loss = vae_loss(x_recon, batch_X, proportions, batch_y, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        recon_loss_epoch += recon_loss.item()
        kl_loss_epoch += kl_loss.item()
        prop_loss_epoch += prop_loss.item()

    total_loss_epoch /= len(train_loader)
    recon_loss_epoch /= len(train_loader)
    kl_loss_epoch /= len(train_loader)
    prop_loss_epoch /= len(train_loader)

    model.eval()
    with torch.no_grad():
        x_val_recon, val_preds_t, mu_t, logvar_t = model(X_val_t.to(device))
        val_preds = val_preds_t.detach().cpu().numpy()
        val_loss, _, _, _ = vae_loss(
            x_val_recon,
            X_val_t.to(device),
            val_preds_t,
            y_val_t.to(device),
            mu_t,
            logvar_t,
        )
        val_loss = val_loss.item()

        val_corrs = []
        for i in range(n_cell_types):
            corr, _ = spearmanr(y_val_props[:, i], val_preds[:, i])
            val_corrs.append(corr)
        avg_val_corr = np.mean(val_corrs)

    scheduler.step(val_loss)

    if progress_enabled:
        try:
            epoch_iter.set_postfix(total=f"{total_loss_epoch:.4f}", recon=f"{recon_loss_epoch:.4f}", kl=f"{kl_loss_epoch:.4f}", prop=f"{prop_loss_epoch:.4f}", corr=f"{avg_val_corr:.3f}")
        except Exception:
            pass

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:6d} {total_loss_epoch:12.6f} {recon_loss_epoch:10.4f} {kl_loss_epoch:10.4f} {prop_loss_epoch:10.4f} {avg_val_corr:12.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        outdir = os.getenv('DECONOMIX_OUTDIR', '.')
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(outdir, 'best_model_vae.pth'))
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

outdir = os.getenv('DECONOMIX_OUTDIR', '.')
model.load_state_dict(torch.load(os.path.join(outdir, 'best_model_vae.pth')))

print("\n" + "="*70)
print("FINAL EVALUATION - VAE")
print("="*70)

model.eval()
with torch.no_grad():
    _, test_predictions_t, test_mu, test_logvar = model(X_test_t.to(device))
    test_predictions = test_predictions_t.detach().cpu().numpy()

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
    ax.scatter(y_test_props[:, i], test_predictions[:, i], alpha=0.5, s=30, c='mediumpurple')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)

    ax.set_xlabel('True Proportion', fontsize=10)
    ax.set_ylabel('Predicted Proportion', fontsize=10)
    ax.set_title(f'{cell_types[i]}\nρ = {correlations[i]:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)

for idx in range(len(cell_types), 9):
    axes[idx].axis('off')

plt.suptitle(f'VAE Deconvolution - Avg ρ = {avg_corr:.3f}',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'vae_deconvolution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(outdir, 'vae_deconvolution.png')}")

print("\n[Analyzing latent space...]")
from sklearn.manifold import TSNE

with torch.no_grad():
    latent_z = test_mu.detach().cpu().numpy()

dominant_ct = np.argmax(y_test_props, axis=1)

tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_z)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                     c=dominant_ct, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Dominant Cell Type', ticks=range(len(cell_types)))
plt.title('VAE Latent Space (t-SNE)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'vae_latent_space.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(outdir, 'vae_latent_space.png')}")

perf_df = pd.DataFrame({
    'cell_type': cell_types,
    'spearman_correlation': correlations,
    'mae': mae_scores,
    'mean_proportion': y_test_props.mean(axis=0)
})
perf_df.to_csv(os.path.join(outdir, 'performance_vae.csv'), index=False)

print("\n" + "="*70)
print("VAE COMPLETE")
print("="*70)
print(f"\nFINAL RESULTS:")
print(f"  Average Spearman: {avg_corr:.3f}")
print(f"  Latent dimension: 128")
print(f"  Architecture: Encoder-Decoder with disentangled latent space")

if avg_corr > 0.7:
    print(f"\nEXCELLENT! Breaking 0.7 barrier!")
elif avg_corr > 0.6:
    print(f"\nGOOD! Above 0.6!")
else:
    print(f"\nStill improving...")

print("\n" + "="*70)
