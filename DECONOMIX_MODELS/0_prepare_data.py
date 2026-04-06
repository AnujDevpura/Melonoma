"""
01_prepare_data.py
Run this ONCE to generate and save the training data.
"""
import scanpy as sc
import numpy as np
import os

print("="*60)
print("🛠️ STEP 1: DATA PREPARATION & SIMULATION")
print("="*60)

# 1. LOAD AND FILTER
print("\n[1/4] Loading Atlas Data...")
p = os.path.join('Data','rna_data.h5ad')
adata = sc.read_h5ad(p if os.path.exists(p) else 'rna_data.h5ad')
adata = adata[adata.obs['disease'] == 'control'].copy()

# Filter for Immune Cells
immune_keywords = ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Monocyte', 
                   'DC', 'Plasma cell', 'Mast cell']
immune_cell_types = [ct for ct in adata.obs['cell_type'].unique() 
                     if any(keyword in ct for keyword in immune_keywords)]
adata = adata[adata.obs['cell_type'].isin(immune_cell_types)].copy()

# Filter Genes
min_cells = int(0.05 * adata.n_obs)
sc.pp.filter_genes(adata, min_cells=min_cells)
adata = adata[:, ~adata.var_names.str.startswith(('MT-', 'RPS', 'RPL'))].copy()

print(f"      Cells: {adata.n_obs}, Genes: {adata.n_vars}")
cell_types = sorted(adata.obs['cell_type'].unique())

# 2. CREATE REFERENCE MATRIX
print("\n[2/4] Creating Reference Matrix...")
sc_expression = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
ref_list = []

for ct in cell_types:
    mask = adata.obs['cell_type'] == ct
    ref_list.append(sc_expression[mask].mean(axis=0))

reference_matrix = np.array(ref_list).T
print(f"      Reference shape: {reference_matrix.shape}")

# 3. SIMULATE BULK DATA
def simulate_bulk(adata, n_bulks, seed):
    np.random.seed(seed)
    bulks, props = [], []
    ct_array = adata.obs['cell_type'].values
    
    print(f"      Generating {n_bulks} samples...")
    for i in range(n_bulks):
        if (i+1) % 100 == 0: print(f"      {i+1}/{n_bulks}...", end='\r')
        
        n_cells = np.random.randint(500, 2000)
        idx = np.random.choice(adata.n_obs, n_cells)
        
        # Make Bulk
        bulk = adata.X[idx].sum(axis=0)
        if hasattr(bulk, 'getA1'): bulk = bulk.getA1() # Handle matrix object
        else: bulk = np.array(bulk).flatten()
        bulks.append(bulk)
        
        # Make Truth
        sampled_cts = ct_array[idx]
        unique, counts = np.unique(sampled_cts, return_counts=True)
        counts_dict = dict(zip(unique, counts/n_cells))
        props.append([counts_dict.get(ct, 0.0) for ct in cell_types])
        
    return np.array(bulks).T, np.array(props).T

print("\n[3/4] Simulating Data (This takes time)...")
train_bulks, train_props = simulate_bulk(adata, n_bulks=800, seed=42)
test_bulks, test_props   = simulate_bulk(adata, n_bulks=300, seed=99)
print("\n      Done.")

# 4. SAVE EVERYTHING
print("\n[4/4] Saving to disk...")
os.makedirs('Processed_Data', exist_ok=True)

np.save('Processed_Data/X_train.npy', train_bulks)       # (Genes, Samples)
np.save('Processed_Data/C_train.npy', train_props)       # (CellTypes, Samples)
np.save('Processed_Data/X_test.npy',  test_bulks)        # (Genes, Samples)
np.save('Processed_Data/C_test.npy',  test_props)        # (CellTypes, Samples)
np.save('Processed_Data/Y_ref.npy',   reference_matrix)  # (Genes, CellTypes)
np.save('Processed_Data/cell_types.npy', cell_types)

print("\n✅ DATA SAVED to 'Processed_Data/' folder!")