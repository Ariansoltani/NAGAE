
# NAGAE: Node Anomaly Graph Autoencoder for Micro-Scale Urban Resilience assessment 🏢

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Geometric](https://img.shields.io/badge/PyG-2.4.0-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the framework presented in:  
**"NAGAE: An Open-Source GNN Framework for Micro-Scale Urban Resilience Modeling Enhanced by Foundation Models"**

---

## 📖 Project Overview
NAGAE is an advanced, unsupervised deep learning framework designed for **Micro-Scale Urban Resilience Assessment**. It addresses the challenge of data scarcity in seismic risk modeling by fusing **15 explicit engineering features** with **64 high-level semantic embeddings** from the **AlphaEarth foundation model**.

The framework models the city as a continuous spatial graph where each parcel is a node, capturing complex spatial interdependencies that traditional IID-based models miss.

### 🧠 Core Architecture (NAGAE-Advanced)
- **Encoder:** A hybrid 3-layer architecture using `GraphSAGE` for inductive neighborhood aggregation and `GAT (Graph Attention Networks)` for anisotropic relationship modeling.
- **Residual Connections:** Implements skip connections to prevent over-smoothing in deep graph structures.
- **Anomaly Detection:** Reconstruction Error from the decoder acts as a direct proxy for physical/morphological vulnerability.
- **Latent Space:** Learns a 32-dimensional "Resilience Fingerprint" for every urban parcel.

---

## 🚀 Technical Workflow (Notebook Pipeline)

The implementation follows a rigorous 14-cell pipeline as demonstrated in the provided `Soltani_NAGAE.ipynb`:

1.  **Environment Setup:** Installing PyTorch Geometric suite and clustering dependencies.
2.  **Preprocessing:** Ordinal strength encoding for structural types and `MinMaxScaler` normalization.
3.  **Graph Construction:** Building a spatial graph using **KNN ($K=6$)** based on Euclidean distances between parcel centroids.
4.  **Model Training:** Unsupervised training with `AdamW` optimizer and `EarlyStopping` to prevent overfitting.
5.  **Resilience Scoring:** Extracting the **R-Score** by synthesizing PCA-derived structural components (PC1) and reconstruction error.
6.  **Typological Clustering:** 
    - Dimensionality reduction via **UMAP** (to 10D for precision).
    - **K-Means clustering** to identify 4 behavioral patterns.
    - **Spatial Smoothing:** Applying **Majority Vote Smoothing ($K=45$)** to eliminate spatial noise and create operational zones.
7.  **Validation:** Statistical validation via **ANOVA**, **Silhouette scores**, and **Random Forest** feature importance.

---

## 📦 Requirements & Installation

The code is optimized for **CUDA 12.1** and **PyTorch 2.5.1**.

```bash
# Core Dependencies
pip install torch==2.5.1+cu121 torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install umap-learn hdbscan scikit-learn pandas seaborn
```

---

## 📊 Key Results & Visualization
The framework generates high-resolution (600 DPI) publication-ready plots:
- **Resilience Methodology Dashboard:** PCA variance and weight sensitivity analysis.
- **Cluster Feature Profiles:** Radar charts showing the "Structural DNA" of each urban typology.
- **Feature Correlation Analysis:** Understanding the interplay between socio-economic and geophysical drivers.
- **Operational Zoning Maps:** Final smoothed maps for municipal intervention.

---

## 📂 Repository Contents
- `Soltani_NAGAE.ipynb`: Full implementation in Jupyter/Colab format.
- `data/`: Sample dataset (Node features and coordinates).
- `results/`: Output Excel files (`final_results.xlsx`) and high-res JPEG visualizations.

## 📝 Citation
If you use this framework or the NAGAE architecture in your research, please cite:
```bibtex
@article{Soltani2026NAGAE,
  title={NAGAE: An Open-Source GNN Framework for Micro-Scale Urban Resilience Modeling Enhanced by Foundation Models},
  author={Ali Madad Soltani, Arian  and Tafti, Mojgan Taheri},
  journal={Computers, Environment and Urban Systems},
  year={2026},
  note={Under Review}
}
```

## 📧 Contact
**Arian Ali Madad Soltani** - [ariansoltani@ut.ac.ir](mailto:ariansoltani@ut.ac.ir)  
*University of Tehran, School of Urban Planning*
