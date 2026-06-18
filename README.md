# NAGAE: Node Anomaly Graph Autoencoder for Micro-Scale Urban Resilience Assessment 🏢

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Geometric](https://img.shields.io/badge/PyG-2.4.0-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the computational framework presented in:  
**"NAGAE: An Open-Source GNN Framework for Micro-Scale Urban Resilience Modeling Enhanced by Foundation Models"**

---

## 📖 Project Overview
NAGAE is an advanced, unsupervised deep learning framework designed for **Micro-Scale Urban Resilience Assessment**. It addresses the challenge of data scarcity in seismic risk modeling by fusing **15 explicit engineering features** with **64 high-level semantic embeddings** extracted from the **AlphaEarth foundation model**.

The framework models the city as a continuous spatial graph where each parcel is a node, capturing complex spatial interdependencies that traditional Independent and Identically Distributed (IID) models miss.

### 🧠 Core Architecture (NAGAE-Advanced)
- **Encoder:** A hybrid 3-layer architecture utilizing `GraphSAGE` for inductive neighborhood aggregation and `GAT` (Graph Attention Networks) for anisotropic relationship modeling.
- **Residual Connections:** Implements skip connections to prevent over-smoothing in deep graph structures.
- **Anomaly Detection:** The Reconstruction Error from the decoder acts as a direct proxy for physical and morphological vulnerability.
- **Latent Space:** Learns a compressed 32-dimensional "Resilience Fingerprint" for every urban parcel.

---

## 🚀 Quick Start & Reproducibility

To ensure full transparency and academic reproducibility, this repository provides a complete, worked tutorial via a Jupyter Notebook (`Soltani_NAGAE.ipynb`) alongside a sample dataset. 

### 1. Requirements & Installation
The code is optimized for **CUDA 12.1** and **PyTorch 2.5.1**. You can replicate the environment using the following commands:

    # Install Core Dependencies
    pip install torch==2.5.1+cu121 torch-geometric
    pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
    pip install umap-learn hdbscan scikit-learn pandas seaborn

### 2. Running the Pipeline
Simply open `Soltani_NAGAE.ipynb` and run all cells. The pipeline follows a rigorous 14-cell workflow:
1. **Preprocessing:** Ordinal strength encoding for structural types and `MinMaxScaler` normalization.
2. **Graph Construction:** Building a spatial graph using **KNN (K=6)** based on Euclidean distances between parcel centroids.
3. **Model Training:** Unsupervised training with `AdamW` optimizer and `EarlyStopping` to prevent overfitting.
4. **Resilience Scoring:** Extracting the **R-Score** by synthesizing PCA-derived structural components (PC1) and reconstruction error.
5. **Typological Clustering:** Dimensionality reduction via **UMAP** (to 10D) followed by **K-Means clustering** to identify 4 behavioral patterns.
6. **Spatial Smoothing:** Applying **Majority Vote Smoothing (K=45)** to eliminate spatial noise and create contiguous operational zones.
7. **Validation:** Statistical validation via **ANOVA**, **Silhouette scores**, and **Random Forest** feature importance.

---

## 📊 Key Results & Visualization
Running the notebook will generate high-resolution, publication-ready plots directly in the output folder, including:
- **Resilience Methodology Dashboard:** PCA variance and weight sensitivity analysis.
- **Cluster Feature Profiles:** Radar charts illustrating the "Structural DNA" of each urban typology.
- **Feature Correlation Analysis:** Heatmaps detailing the interplay between socio-economic and geophysical drivers.

---

## 📂 Repository Contents
- `Soltani_NAGAE.ipynb`: Full, reproducible implementation in Jupyter Notebook format.
- `data/`: Sample geospatial dataset (node features, coordinates, and AlphaEarth embeddings).
- `results/`: Directory for output Excel files (`final_results.xlsx`) and generated visualizations.

---

## 📝 Citation
If you utilize this framework or the NAGAE architecture in your research, please cite our paper:


