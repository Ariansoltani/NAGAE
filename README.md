```markdown
# NAGAE: Node Anomaly Graph Autoencoder for Urban Resilience 🌱🏢

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: CEUS (Under Review)](https://img.shields.io/badge/Paper-CEUS-green.svg)](#)

> Official implementation of the paper: **"A Graph Neural Network Framework for Micro-Scale Urban Resilience Assessment Against Seismic Risk"** (Submitted to *Computers, Environment and Urban Systems*).

## 📖 Overview
**NAGAE (Node Anomaly Graph Autoencoder)** is an open-source, unsupervised deep learning framework designed to assess urban seismic resilience at the micro-scale (building parcel level) in data-scarce environments. 

By synergizing **15 explicit engineering features** with **64 high-level semantic embeddings** extracted from the **AlphaEarth** foundation model, NAGAE processes the city as a continuous spatial graph. It overcomes traditional Euclidean limitations and introduces a context-aware approach to quantify structural vulnerability and urban morphology without relying on historical damage data.

### ✨ Key Features:
- **Hybrid Graph Architecture:** Fuses GraphSAGE (for inductive efficiency) and GAT (for attention mechanisms) with residual skip connections.
- **Unsupervised Anomaly Detection:** Extracts a 32-dimensional "Resilience Fingerprint" and uses reconstruction error as a direct proxy for physical vulnerability.
- **Multi-faceted Outputs:** 
  1. A continuous **Resilience Score (R-Score)** via PCA.
  2. **Behavioral Typologies** via UMAP and K-Means clustering.
- **Data-Scarce Ready:** Entirely eliminates the need for historical seismic labeled data.

---

## 🏗️ Architecture
![NAGAE Architecture]([Link_to_your_architecture_image_e.g._Figure_7_in_repo])
*The NAGAE framework: (Left) Advanced Geometric Encoder, (Middle) Latent Space Representation, (Right) Decoder & Anomaly Detection Strategy.*

---

## 📂 Repository Structure
```text
📦 NAGAE
 ┣ 📂 data
 ┃ ┣ 📜 sample_nodes.csv       # Sample dataset containing 79D features
 ┃ ┗ 📜 sample_edges.csv       # Graph edge index (Spatial KNN)
 ┣ 📂 models
 ┃ ┣ 📜 nagae.py               # Core PyTorch Geometric model (SAGE + GAT)
 ┃ ┗ 📜 layers.py              # Custom graph layers and attention heads
 ┣ 📂 utils
 ┃ ┣ 📜 graph_builder.py       # Constructs the spatial graph (K=6)
 ┃ ┗ 📜 metrics.py             # PCA, UMAP, and R-Score calculation
 ┣ 📜 train.py                 # Training script for the Autoencoder
 ┣ 📜 cluster.py               # Generates resilience typologies
 ┣ 📜 requirements.txt         # Dependencies
 ┗ 📜 README.md
```

---

## ⚙️ Installation & Requirements

The framework is built using PyTorch and PyTorch Geometric. It is optimized for parallel GPU processing via CUDA 12.1.

1. Clone the repository:
```bash
git clone https://github.com/Ariansoltani/NAGAE.git
cd NAGAE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure you have the correct version of `torch-scatter` and `torch-sparse` matching your CUDA version).*

---

## 🚀 Quick Start

### 1. Data Preparation
Place your urban dataset in the `data/` directory. The framework expects a node feature matrix (79 dimensions: 15 engineered + 64 AlphaEarth embeddings) and spatial coordinates to build the KNN graph. A `sample_dataset` is provided for testing.

### 2. Train the NAGAE Model
Run the unsupervised training pipeline to learn latent embeddings and calculate reconstruction errors:
```bash
python train.py --epochs 200 --k_neighbors 6 --hidden_dim 128 --latent_dim 32
```

### 3. Extract Typologies & R-Score
Extract the composite R-Score and cluster the parcels into 4 distinct behavioral typologies:
```bash
python cluster.py --n_clusters 4
```

---

## 📊 Outputs
The model generates the following outputs in the `results/` folder:
- `resilience_scores.csv`: Contains the PCA-derived R-Score for each parcel.
- `typologies.csv`: Cluster assignments (0 to 3) based on UMAP + K-Means.
- `latent_embeddings.npy`: The 32D compressed representation for advanced spatial analysis.

---

## 📝 Citation
If you find this code or our conceptual framework (e.g., *Spatial Decoupling*, *Fabric Lock-in*) useful in your research, please cite our paper:

```bibtex
@article{Soltani2024NAGAE,
  title={A Graph Neural Network Framework for Micro-Scale Urban Resilience Assessment Against Seismic Risk},
  author={Soltani, Arian Ali Madad and Tafti, Mojgan Taheri},
  journal={Computers, Environment and Urban Systems},
  year={2024},
  note={Under Review}
}
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📫 Contact
For questions or collaborations, feel free to open an issue or contact: **ariansoltani@ut.ac.ir**
```
