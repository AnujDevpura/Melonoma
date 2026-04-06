import scanpy as sc
import pandas as pd
import os

# Load the H5AD file
print("Loading skin atlas data...")
p = os.path.join('Data','rna_data.h5ad')
adata = sc.read_h5ad(p if os.path.exists(p) else 'rna_data.h5ad')  

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)
print(adata)
print(f"\nTotal cells: {adata.n_obs:,}")
print(f"Total genes: {adata.n_vars:,}")

print("\n" + "=" * 70)
print("CELL METADATA (adata.obs columns)")
print("=" * 70)
print(adata.obs.columns.tolist())

print("\n" + "=" * 70)
print("FIRST FEW ROWS OF METADATA")
print("=" * 70)
print(adata.obs.head(10))

print("\n" + "=" * 70)
print("LOOKING FOR CELL TYPE INFORMATION")
print("=" * 70)

# Common cell type column names
possible_celltype_cols = [
    'cell_type', 'celltype', 'Cell_type', 'CellType',
    'celltype_major', 'celltype_minor', 
    'cell_ontology_class', 'annotation',
    'cluster', 'clusters', 'leiden', 'louvain',
    'cell_identity', 'predicted_labels'
]

found_celltype_col = None
for col in possible_celltype_cols:
    if col in adata.obs.columns:
        found_celltype_col = col
        print(f"✅ Found cell type column: '{col}'")
        print(f"\nNumber of cell types: {adata.obs[col].nunique()}")
        print(f"\nCell type distribution:")
        print(adata.obs[col].value_counts())
        break

if not found_celltype_col:
    print("❌ No obvious cell type column found")
    print("Available columns:", adata.obs.columns.tolist())

print("\n" + "=" * 70)
print("LOOKING FOR DISEASE/SAMPLE INFORMATION")
print("=" * 70)

# Check for disease column
disease_cols = ['disease', 'Disease', 'condition', 'sample_type', 'tissue_type']
for col in disease_cols:
    if col in adata.obs.columns:
        print(f"\n✅ Found disease column: '{col}'")
        print(adata.obs[col].value_counts())
        break

# Check for sample IDs
sample_cols = ['sample', 'sample_id', 'Sample', 'batch', 'patient']
for col in sample_cols:
    if col in adata.obs.columns:
        print(f"\n✅ Found sample column: '{col}'")
        print(f"Number of samples: {adata.obs[col].nunique()}")
        print(f"Samples: {adata.obs[col].unique()[:10]}")  # First 10
        break

print("\n" + "=" * 70)
print("GENE INFORMATION")
print("=" * 70)
print(adata.var.columns.tolist())
print(f"\nFirst 10 genes:")
print(adata.var_names[:10].tolist())

print("\n" + "=" * 70)
print("CHECKING FOR MELANOMA SAMPLES")
print("=" * 70)

# Search for melanoma in any text column
melanoma_found = False
for col in adata.obs.columns:
    if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
        melanoma_mask = adata.obs[col].astype(str).str.contains('melanoma', case=False, na=False)
        if melanoma_mask.any():
            melanoma_found = True
            n_melanoma = melanoma_mask.sum()
            print(f"✅ Found {n_melanoma:,} melanoma cells in column '{col}'!")
            print(f"\nValues containing 'melanoma':")
            print(adata.obs.loc[melanoma_mask, col].value_counts())

if not melanoma_found:
    print("⚠️ No explicit 'melanoma' mention found")
    print("The dataset might contain melanoma under a different label")
    print("or might be only healthy skin samples")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total cells: {adata.n_obs:,}")
print(f"Total genes: {adata.n_vars:,}")
if found_celltype_col:
    print(f"Cell types: {adata.obs[found_celltype_col].nunique()}")
else:
    print("Cell types: Not clearly labeled")