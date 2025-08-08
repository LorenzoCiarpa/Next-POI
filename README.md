# Next Point of Interest Prediction with Graph Neural Networks and User Clustering 🗺️🤖

An advanced machine learning system for predicting next Points of Interest (POI) using Graph Neural Networks enhanced with user behavior clustering, implementing both the baseline HMT-GRN model and our novel HMT-GRN-C (Clustering) variant.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Models](#models)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)

---

## Overview

This research project addresses the challenge of predicting where users will visit next based on their historical mobility patterns. The system leverages temporal and spatial information to make accurate POI predictions, with our key contribution being the integration of user behavior clustering to enhance prediction accuracy.

### 🎯 **Research Objectives**
- **Temporal Modeling**: Capture temporal dependencies in user mobility patterns
- **Spatial Relationships**: Model geographical relationships between POIs
- **User Behavior Analysis**: Cluster users based on mobility patterns to improve personalization
- **Enhanced Predictions**: Combine clustering insights with Graph Neural Networks for superior performance

### 🔬 **Key Contributions**
- **Novel Clustering Integration**: Enhanced HMT-GRN with user behavior clustering (HMT-GRN-C)
- **Multi-granular Spatial Modeling**: GeoHash encoding at multiple resolutions (2-6 levels)
- **Temporal-Spatial Graph Attention**: Dual attention mechanisms for comprehensive modeling
- **Comprehensive Evaluation**: Extensive experiments on real-world location datasets

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Next POI Prediction System                       │
├─────────────────────────────────────────────────────────────────────┤
│                        Input Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ POI Sequences   │  │ User Trajectories│  │ Temporal Features   │  │
│  │ • Historical    │  │ • User IDs       │  │ • Time Deltas       │  │
│  │   Visits        │  │ • Mobility       │  │ • Sequence Order    │  │
│  │ • Location IDs  │  │   Patterns       │  │ • Timestamps        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Feature Embedding Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ POI Embeddings  │  │ GeoHash Embed.  │  │ User Embeddings     │  │
│  │ • Dense Vector  │  │ • Multi-level   │  │ • User Profiles     │  │
│  │   Representation│  │   (2,3,4,5,6)   │  │ • Behavior Vectors  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      Clustering Module (HMT-GRN-C)                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    User Behavior Clustering                     │ │
│  │  ┌─────────────────┐    ┌─────────────────┐                    │ │
│  │  │ Feature         │    │ Clustering      │                    │ │
│  │  │ Extraction      │    │ Algorithms      │                    │ │
│  │  │ • Mobility      │    │ • K-Means       │                    │ │
│  │  │   Patterns      │────│ • Gaussian      │                    │ │
│  │  │ • Visit         │    │   Mixture       │                    │ │
│  │  │   Frequency     │    │ • Custom        │                    │ │
│  │  │ • Spatial       │    │   Distance      │                    │ │
│  │  │   Preferences   │    │   Metrics       │                    │ │
│  │  └─────────────────┘    └─────────────────┘                    │ │
│  │           │                       │                            │ │
│  │           └───────────────────────┼────────────────────────────┘ │
│  │                                   ▼                              │
│  │                        Cluster Embeddings                       │
├─────────────────────────────────────────────────────────────────────┤
│                     Graph Neural Network Layer                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Dual Attention Mechanism                     │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────┐   │ │
│  │  │    Temporal Graph       │  │    Spatial Graph            │   │ │
│  │  │    Attention (GAT)      │  │    Attention (GAT)          │   │ │
│  │  │  ┌─────────────────┐    │  │  ┌─────────────────────┐    │   │ │
│  │  │  │ Sequential      │    │  │  │ Geographic          │    │   │ │
│  │  │  │ Dependencies    │    │  │  │ Relationships       │    │   │ │
│  │  │  │ • Time-aware    │    │  │  │ • Distance-based    │    │   │ │
│  │  │  │ • Visit Order   │    │  │  │ • GeoHash           │    │   │ │
│  │  │  │ • Recency       │    │  │  │   Hierarchy         │    │   │ │
│  │  │  └─────────────────┘    │  │  └─────────────────────┘    │   │ │
│  │  └─────────────────────────┘  └─────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      Sequence Modeling Layer                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                        LSTM Network                             │ │
│  │  • Processes attended temporal-spatial features                 │ │
│  │  • Captures long-term sequential dependencies                   │ │
│  │  • Generates contextual representations                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Fusion Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   Feature Fusion & Prediction                   │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │ │
│  │  │ LSTM Features   │  │ User Embeddings │  │ Cluster         │  │ │
│  │  │ • Sequential    │  │ • User Profiles │  │ Embeddings      │  │ │
│  │  │   Context       │─►│ • Personal      │─►│ (HMT-GRN-C)     │  │ │
│  │  │ • Temporal      │  │   Preferences   │  │ • Group         │  │ │
│  │  │   Patterns      │  │ • Behavior      │  │   Behavior      │  │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Output Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Multi-level Predictions                      │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │ │
│  │  │ Next POI        │  │ GeoHash         │  │ Top-K           │  │ │
│  │  │ Prediction      │  │ Predictions     │  │ Ranking         │  │ │
│  │  │ • Exact POI     │  │ • Multi-level   │  │ • Ranked List   │  │ │
│  │  │ • Probability   │  │   (2,3,4,5,6)   │  │ • Confidence    │  │ │
│  │  │   Distribution  │  │ • Hierarchical  │  │   Scores        │  │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Models

### 📊 **HMT-GRN (Baseline Model)**

**Hierarchical Multi-Task Graph Recurrent Network**
- **Temporal Graph Attention**: Models sequential dependencies in user visits
- **Spatial Graph Attention**: Captures geographical relationships between POIs
- **Multi-task Learning**: Predicts POIs at multiple GeoHash granularities
- **LSTM Integration**: Processes attended features for sequence modeling

**Architecture Components**:
```python
# Core model structure
class hmt_grn(nn.Module):
    - temporalGAT: Multi-head attention for temporal relationships
    - spatialGAT: Multi-head attention for spatial relationships  
    - poiEmbed: POI embedding layer
    - geoHashEmbed[2-6]: Multi-level GeoHash embeddings
    - userEmbed: User profile embeddings
    - ownLSTM: Custom LSTM for sequence processing
    - fuseDense: Final prediction layer
```

### 🎯 **HMT-GRN-C (Our Enhanced Model)**

**HMT-GRN with Clustering Enhancement**
- **All HMT-GRN Features**: Inherits full baseline functionality
- **Cluster Embeddings**: Additional embedding layer for user clusters
- **Enhanced Fusion**: Modified fusion layer incorporating cluster information
- **Improved Prediction**: Better performance through behavioral grouping

**Key Enhancements**:
```python
# Additional components in HMT-GRN-C
class hmt_grn(nn.Module):
    - clustersEmbed: Cluster embedding layer
    - fuseDenseClusters: Enhanced fusion with cluster features
    # Additional cluster processing in forward pass
```

### 🏆 **Model Variants**

1. **Standard Models** (`model.py`, `train.py`, `test.py`)
   - Baseline HMT-GRN implementation
   - Standard evaluation metrics

2. **Clustering Enhanced** (`model_clusters.py`, `train_clusters.py`, `test_clusters.py`)
   - Our novel HMT-GRN-C model
   - Integrated cluster-based features

3. **Top-K Variants** (`model_top_k.py`, `train_top_k.py`, `test_top_k.py`)
   - Specialized for ranking evaluation
   - Top-K accuracy optimization

---

## Dataset

### 📍 **Supported Datasets**

- **Gowalla**: Large-scale location-based social network dataset

### 📊 **Data Preprocessing**

- **Trajectory Segmentation**: Split user trajectories into training sequences
- **Temporal Features**: Extract time deltas and visit patterns
- **Spatial Encoding**: Generate multi-level GeoHash representations

### 🔄 **Data Format**

```python
# Input data structure
{
    'pois_seq': [[poi1, poi2, poi3, ...], ...],      # POI sequences
    'delta_t_seq': [[dt1, dt2, dt3, ...], ...],      # Time deltas
    'delta_d_seq': [[dd1, dd2, dd3, ...], ...],      # Distance deltas
    'users': [[u1, u2, u3, ...], ...],               # User sequences
    'geohash_[2-6]': [...],                          # Multi-level GeoHash
    'clusters': {...}                                 # Cluster assignments
}
```

---

## Installation

### Prerequisites

```bash
# Core dependencies
torch>=1.8.0
torch-geometric>=2.0.0
numpy>=1.19.0
scikit-learn>=0.24.0
networkx>=2.5
tqdm>=4.60.0
pickle5>=0.0.11
```

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/LorenzoCiarpa/Next-POI.git
   cd Next-POI
   ```

2. **Install dependencies**
   ```bash
   pip install torch torch-geometric numpy scikit-learn networkx tqdm
   ```

3. **Prepare data**
   ```bash
   # Place your dataset files in the data/ directory
   mkdir -p data/processedFiles
   ```

4. **Generate clusters** (for HMT-GRN-C model)
   ```bash
   python clustering.py
   ```

---

## Usage

### 🚂 **Training Models**

#### **Baseline HMT-GRN**
```bash
cd train
python train.py
```

#### **Enhanced HMT-GRN-C (with clustering)**
```bash
cd train
python train_clusters.py
```

#### **Top-K Variant**
```bash
cd train
python train_top_k.py
```

### 🧪 **Testing Models**

#### **Standard Evaluation**
```bash
cd tests
python test.py               # Baseline model
python test_clusters.py      # Clustering enhanced
python test_top_k.py         # Top-K evaluation
```

### 🔬 **Clustering Analysis**

Generate and analyze user behavior clusters:

```bash
python clustering.py
```

**Clustering Options**:
- **K-Means**: Traditional distance-based clustering
- **Gaussian Mixture**: Probabilistic clustering with soft assignments
- **Custom Metrics**: Domain-specific distance functions

### ⚙️ **Configuration**

Key hyperparameters in training scripts:

```python
arg = {
    'epoch': 20,                    # Training epochs
    'embedding_dim': 1024,          # Embedding dimensions
    'hidden_dim': 1024,             # Hidden layer size
    'userEmbed_dim': 256,           # User embedding size
    'beamSize': 100,                # Beam search size
    'dropout': 0.1,                 # Dropout rate
    'learning_rate': 0.001,         # Learning rate
    'max_length': 50,               # Max sequence length
}
```

---


## Project Structure

```
Next-POI/
├── README.md                   # Project documentation
├── EAI_Report.pdf             # Detailed technical report
├── clustering.py              # User behavior clustering implementation
├── data/                      # Dataset directory
│   ├── dataset_view.py        # Data exploration utilities
│   └── processedFiles/        # Processed dataset files
├── models/                    # Model implementations
│   ├── model.py               # Baseline HMT-GRN model
│   ├── model_clusters.py      # Enhanced HMT-GRN-C model
│   └── model_top_k.py         # Top-K specialized variant
├── train/                     # Training scripts
│   ├── train.py               # Baseline model training
│   ├── train_clusters.py      # Clustering model training
│   └── train_top_k.py         # Top-K model training
├── tests/                     # Evaluation scripts
│   ├── test.py                # Baseline model evaluation
│   ├── test_clusters.py       # Clustering model evaluation
│   └── test_top_k.py          # Top-K model evaluation
└── utils/                     # Utility functions
    ├── func.py                # General utilities
    ├── func_clusters.py       # Clustering-specific utilities
    └── func_top_k.py          # Top-K utilities
```

For detailed methodology, experimental setup, and results, please refer to our comprehensive technical report: `EAI_Report.pdf`

---


## License

This project is released under [License Type] for academic and research purposes.

---


**Note**: For complete technical details, experimental setup, and comprehensive results analysis, please refer to the included technical report: `EAI_Report.pdf`