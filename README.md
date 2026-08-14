
```markdown
# NAGAE: Node Attributes-focused Graph AutoEncoder for Micro-Scale Urban Resilience 🏢

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![PyG 2.8.0](https://img.shields.io/badge/PyG-2.8.0-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official open-source implementation of the computational framework presented in:  
**"Unsupervised Spatial Representation Learning for Micro-Scale Urban Resilience Assessment: A Case Study of Tehran"**

---

## 📖 Project Overview

**NAGAE** is an advanced, unsupervised graph-based deep learning framework designed for **Micro-Scale Urban Resilience Assessment**. It addresses the critical challenge of data scarcity in seismic risk modeling by fusing **15 explicit engineering indicators** with **64 high-level visual-semantic embeddings** extracted from the **AlphaEarth foundation model**.

Moving beyond the traditional Independent and Identically Distributed (IID) assumption, NAGAE models the urban fabric as a continuous spatial graph. By treating individual parcels as interconnected nodes, the framework captures complex spatial interdependencies, topological constraints, and localized morphological decay that conventional distance-based GIS models overlook.

### 🧠 Core Architecture (NAGAE-Advanced)
- **Hybrid Encoder:** A 3-layer architecture fusing `GraphSAGE` (for inductive, scalable neighborhood aggregation) and `GAT` (Graph Attention Networks for anisotropic, dynamic relationship weighting).
- **Residual Skip Connections:** Prevents topological over-smoothing and preserves individual building identities during deep graph message-passing.
- **Dual-Signal Anomaly Detection:** Utilizes reconstruction error strictly as a *secondary contextual novelty signal*, combined with the primary structural component (PC1) to prevent the false penalization of "positive anomalies" (e.g., modern, resilient buildings embedded in decayed fabrics).
- **Latent Resilience Fingerprint:** Compresses the 79-dimensional input space into a highly cohesive 32-dimensional latent manifold.
- **Dual-Level Explainability:** Intrinsic interpretability via GAT attention coefficients, complemented by post-hoc policy transparency using Random Forest feature importance.

---

## 🚀 Quick Start & Reproducibility

To ensure full academic transparency, this repository provides a complete, reproducible pipeline via a Jupyter Notebook alongside a representative sample dataset. 

### 1. Requirements & Installation
The codebase is optimized for **CUDA-enabled GPUs** and requires **Python 3.12+**, **PyTorch 2.5.1**, and **PyTorch Geometric 2.8.0**.

```bash
# Install Core Dependencies
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.8.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Install Spatial & Analytical Libraries
pip install umap-learn scikit-learn pandas seaborn matplotlib openpyxl
```

### 2. Running the Pipeline
Open `Soltani_NAGAE.ipynb` and execute the cells. The pipeline follows a rigorous, multi-stage workflow:

1. **Data Preprocessing:** Imputation of missing telemetry via neighborhood medians, neutralization of severe outliers using a `RobustScaler` (IQR-based), and ordinal encoding of structural types based on material strength logic.
2. **Physical Graph Construction:** Generation of an undirected spatial graph using an inductive **Spatial KNN (K=6)** algorithm. The ~38m Euclidean radius is physically grounded to capture immediate seismic coupling mechanisms (e.g., structural pounding and wave incoherence).
3. **Unsupervised Representation Learning:** Full-batch training of the NAGAE architecture using the `Adam` optimizer and Mean Squared Error (MSE) loss, governed by `EarlyStopping` and `ReduceLROnPlateau` schedulers.
4. **Resilience Scoring (R-Score):** Synthesis of the continuous R-Score via percentile-based rank fusion of the dominant PCA structural component ($\alpha=0.7$) and the inverse reconstruction error ($\beta=0.3$).
5. **Typological Clustering:** Non-linear manifold reduction via **UMAP** (32D $\to$ 10D) to preserve topological nuances, followed by **K-Means clustering (K=4)** to extract distinct behavioral urban typologies.
6. **Operational Zoning:** Application of **Majority Vote Smoothing (K=45)** to bridge the micro-to-meso scale transition, eliminating spatial noise to generate contiguous, actionable planning zones.
7. **Validation & Interpretability:** Statistical validation via one-way ANOVA and Silhouette analysis, alongside post-hoc semantic alignment using Pearson correlation and Random Forest feature importance.

---

## 📊 Key Outputs & Visualizations

Executing the notebook will generate high-resolution, publication-ready analytical dashboards in the `results/` directory:
- **Resilience Methodology Dashboard:** PCA variance decomposition, weight sensitivity analysis, and internal correlation plots.
- **Cluster Feature Profiles (Radar Charts):** Visualizing the "Structural DNA" of the four identified typologies (Planned Modern, Consolidated Residential, Topographically Constrained, and Critical Dilapidated).
- **Semantic Correlation Heatmaps:** Detailing the informational independence and complementary synergy between explicit engineering features and AlphaEarth visual embeddings.
- **Operational Zoning Maps:** Geospatial representations of contiguous urban blocks prioritized for municipal intervention.

---

## 📂 Repository Structure

```text
├── Soltani_NAGAE.ipynb      # Full, reproducible end-to-end implementation
├── data/
│   ├── sample_parcels.csv   # Representative geospatial dataset (coordinates + 15 engineered features)
│   └── alphaearth_embeds.npy# Sample 64-dimensional foundation model embeddings
├── results/                 # Auto-generated output directory for plots and Excel exports
├── requirements.txt         # Pinned environment dependencies
└── README.md                # Project documentation
```

---

## 📜 Citation

If you use the NAGAE framework, code, or conceptual methodology in your research, please cite our paper:

```bibtex
@article{Soltani2026NAGAE,
  title={Unsupervised Spatial Representation Learning for Micro-Scale Urban Resilience Assessment: A Case Study of Tehran},
  author={ Ali Madad Soltani, Arian},
  journal={Sustainable Cities and Society},
  year={2026},
  publisher={Elsevier}
}
```

## 🤝 Contributing & License
This project is released under the **MIT License**. We welcome contributions, issue reports, and adaptations to other urban biomes. For questions regarding the mathematical formulations or architectural choices, please open an issue on GitHub.
```
