# GEM

This repo provides official code for "Subgraphs as First-Class Citizens in Incident Management for Large-Scale Online Systems: An Evolution-Aware Framework".

# Introduction

This work extends upon our previous publication ``Graph based Incident Extraction and Diagnosis in Large-Scale Online Systems'' at the 37th IEEE/ACM International Conference on Automated Software Engineering (ASE 2022)''.

## Project Structure

- `./data` contains the simulation environment dataset and open-sourced datasets used for helping the understanding and reporduction of each step of GEM.
- `./src` contains the implementation of GEM extracted for reproduction. 
- `./demo` contains ipython notebooks which provide examples to show how each step of GEM is performed. Their order is as follow:
    - `anomaly_detection_and_impact_extraction.ipynb` contains code for telemetry data anomaly detection and impact extraction.
    - `data_labelling.ipynb` contains code for data labelling using fault injection records.
    - `feature_engineering.ipynb` contains code for feature engineering.
    - `incident_detection.ipynb` contains code for the graph neural networks based model training and testing for incident detection on the simulation environment dataset.
    - `incident_diagnosis_using_edge_clues.ipynb` contains code for the incident diagnosis on the simulation environment dataset using edge clues.
    - `incident_diagnosis_using_node_clues_with_continual_optimization_OB.ipynb` contains code for the incident diagnosis on dataset OB using node clues with continual optimization.
    - `incident_diagnosis_using_node_clues_with_continual_optimization_AIOPS2021.ipynb` contains code for the incident diagnosis on dataset AIOPS2021 using node clues with continual optimization.



# Usage

## Prerequisites

The GEM framework has different requirements for incident detection and incident diagnosis components. Install the appropriate dependencies based on your use case:

### For Incident Detection
```bash
pip install -r requirements_for_incident_detection.txt
```

### For Incident Diagnosis
```bash
pip install -r requirements_for_incident_diagnosis.txt
```

## Quick Start

The GEM framework follows a sequential workflow for incident management in large-scale online systems. Follow these steps in order:

### Step 1: Anomaly Detection and Impact Extraction

Start by detecting anomalies in raw telemetry data and extracting their impact:

```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# Load monitoring data
all_data = pd.read_csv("./data/calling_relationships_monitoring.csv")

# Run anomaly detection and impact extraction
# See demo/anomaly_detection_and_impact_extraction.ipynb for detailed implementation
```

### Step 2: Data Labelling

Label the data using historical incident reports or fault injection records:

```python
# Load raw topologies
with open('./data/raw_topoloies.pkl', 'rb') as f:
    Topologies = pickle.load(f)

# Perform data labelling
# See demo/data_labelling.ipynb for detailed implementation
```

### Step 3: Feature Engineering

Preprosses the data and create feature vectors for incident detection:

```python
# Feature engineering for graph-based incident detection
# See demo/feature_engineering.ipynb for detailed implementation
```

### Step 4: Incident Detection

Train and test the graph neural network model for incident detection:

```python
import sys
sys.path.append('./src')
from incident_detection import callSpatioDevNet
import torch
from torch_geometric.data import Data, DataLoader

# Load training and test data
with open('./data/train_cases.pkl', 'rb') as f:
    train_cases = pickle.load(f)
    
with open('./data/test_cases.pkl', 'rb') as f:
    test_cases = pickle.load(f)

# Initialize and train the model
model = callSpatioDevNet.callSpatioDevNet(
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    hidden_dim=20
)

# Train the model
model.fit(train_cases)

# Test the model
results = model.predict(test_cases)
```

### Step 5: Incident Diagnosis

Perform incident diagnosis using either edge clues or node clues:

#### Using Edge Clues
```python
from incident_diagnosis import incident_diagnosis

# Diagnose incidents using edge clues
# See demo/incident_diagnosis_using_edge_clues.ipynb for detailed implementation
```

#### Using Node Clues with Continual Optimization
```python
# For Online Boutique dataset
# See demo/incident_diagnosis_using_node_clues_with_continual_optimization_OB.ipynb

# For AIOPS2021 dataset
# See demo/incident_diagnosis_using_node_clues_with_continual_optimization_AIOPS2021.ipynb
```

## Detailed Examples

For comprehensive examples and detailed implementations, refer to the Jupyter notebooks in the `./demo` directory:

1. **Anomaly Detection**: `demo/anomaly_detection_and_impact_extraction.ipynb`
2. **Data Labelling**: `demo/data_labelling.ipynb`
3. **Feature Engineering**: `demo/feature_engineering.ipynb`
4. **Incident Detection**: `demo/incident_detection.ipynb`
5. **Incident Diagnosis (Edge Clues)**: `demo/incident_diagnosis_using_edge_clues.ipynb`
6. **Incident Diagnosis (Node Clues - OB)**: `demo/incident_diagnosis_using_node_clues_with_continual_optimization_OB.ipynb`
7. **Incident Diagnosis (Node Clues - AIOPS2021)**: `demo/incident_diagnosis_using_node_clues_with_continual_optimization_AIOPS2021.ipynb`

## Data Structure

The framework works with the following data files:

- `calling_relationships_monitoring.csv`: Monitoring data for service call relationships
- `injected_faults.csv`: Records of injected faults for training
- `platform_faults.csv`: Platform-level fault information
- `raw_topoloies.pkl`: Raw topology data
- `issue_topoloies.pkl`: Processed issue topology data
- `train_cases.pkl` / `test_cases.pkl`: Preprocessed training and testing datasets
- `AIOPS2021.pkl` / `OB.pkl`: Specific datasets for evaluation

## Model Files

Pre-trained models are available in the `./demo` directory:
- `FinalModel_OnlineBoutique.pt`: Trained model for Online Boutique dataset

## API Reference

Below is the detailed API documentation for GEM.


### Incident Detection Module

#### `callSpatioDevNet`

A graph neural network-based model for incident detection using spatio-temporal features.

**Import:**
```python
from src.incident_detection import callSpatioDevNet
```

**Constructor:**
```python
model = callSpatioDevNet(
    name='SpatioDevNetPackage',
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    input_dim=None,
    hidden_dim=20,
    edge_attr_len=60,
    global_fea_len=2,
    num_layers=2,
    edge_module='linear',
    act=True,
    pooling='attention',
    is_bilinear=False,
    nonlinear_scorer=False,
    head=4,
    aggr='mean',
    concat=False,
    dropout=0.4,
    weight_decay=1e-2,
    loss_func='focal_loss',
    seed=None,
    gpu=None,
    ipython=True,
    details=True
)
```

**Key Parameters:**
- `name` (str): Model identifier for saving/loading
- `num_epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `lr` (float): Learning rate for optimization
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `edge_attr_len` (int): Edge attribute length
- `global_fea_len` (int): Global feature length
- `num_layers` (int): Number of GNN layers
- `edge_module` (str): Edge processing module ('linear' or 'lstm')
- `pooling` (str): Graph pooling method ('attention', 'max', 'mean', 'add')
- `loss_func` (str): Loss function ('focal_loss', 'dev_loss', 'cross_entropy')
- `dropout` (float): Dropout rate for regularization
- `seed` (int): Random seed for reproducibility
- `gpu` (int): GPU device ID

**Methods:**

##### `fit(datalist, valid_list=None, log_step=20, patience=10, valid_proportion=0.0, early_stop_fscore=None)`
Train the incident detection model.

**Parameters:**
- `datalist` (list): Training data as PyTorch Geometric Data objects
- `valid_list` (list, optional): Validation dataset
- `log_step` (int): Logging frequency during training
- `patience` (int): Early stopping patience
- `valid_proportion` (float): Proportion of data for validation split
- `early_stop_fscore` (float, optional): F-score threshold for early stopping

##### `predict(datalist)`
Predict anomaly scores for input data.

**Parameters:**
- `datalist` (list): Input data as PyTorch Geometric Data objects

**Returns:**
- `outputs` (numpy.ndarray): Anomaly scores
- `features` (numpy.ndarray): Extracted features

##### `cold_start_predict(datalist, n_neighbors=3)`
Perform prediction with cold start using k-nearest neighbors.

**Parameters:**
- `datalist` (list): Input data
- `n_neighbors` (int): Number of neighbors for KNN

**Returns:**
- `knn_preds` (list): KNN predictions
- `knn_pred_proba` (list): KNN prediction probabilities
- `knn` (object): Trained KNN classifier

##### `load(model_file=None)`
Load a pre-trained model.

**Parameters:**
- `model_file` (str, optional): Path to model file

#### Utility Functions

##### `bf_search(labels, scores)`
Find optimal threshold using binary search for best F1-score.

**Parameters:**
- `labels` (array): True labels
- `scores` (array): Prediction scores

**Returns:**
- `results` (tuple): Precision, recall, F-score metrics
- `threshold` (float): Optimal threshold

### Incident Diagnosis Module

#### Core Functions

**Import:**
```python
from src.incident_diagnosis import incident_diagnosis
```

##### `get_weight_from_edge_info(topology, clue_tag, statistics=None)`
Calculate edge weights from topology information.

**Parameters:**
- `topology` (dict): Network topology with node and edge information
- `clue_tag` (str): Metric tag for weight calculation
- `statistics` (dict, optional): Statistical normalization parameters

**Returns:**
- `weight` (dict): Calculated edge weights

##### `get_anomaly_graph(topology, node_clue_tags=[], edge_clue_tags=[], a=None, get_edge_weight=None, edge_backward_factor=0.3)`
Construct anomaly graph from topology and clues.

**Parameters:**
- `topology` (dict): Network topology data
- `node_clue_tags` (list): Node-level clue tags
- `edge_clue_tags` (list): Edge-level clue tags
- `a` (dict, optional): Rescaling factors for clues
- `get_edge_weight` (function): Edge weight calculation function
- `edge_backward_factor` (float): Backward edge weight factor

**Returns:**
- `anomaly_graph` (networkx.DiGraph): Constructed anomaly graph

##### `root_cause_localization(case, node_clue_tags, edge_clue_tags, a, get_edge_weight=None, edge_backward_factor=0.3)`
Localize root cause using PageRank on anomaly graph.

**Parameters:**
- `case` (dict): Incident case data
- `node_clue_tags` (list): Node clue tags
- `edge_clue_tags` (list): Edge clue tags
- `a` (dict): Clue rescaling factors
- `get_edge_weight` (function, optional): Edge weight function
- `edge_backward_factor` (float): Backward propagation factor

**Returns:**
- `root_cause` (str): Identified root cause node

##### `explain(case, target='root_cause')`
Generate explanations for incident diagnosis.

**Parameters:**
- `case` (dict): Incident case data
- `target` (str): Target field to explain

**Returns:**
- `explanation` (list): Sorted explanation features with power scores

##### `optimize(case, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor, historical_incident_topologies, init_clue_tag, range_a=5, num_trials=100)`
Optimize clue weights using historical data and Optuna.

**Parameters:**
- `case` (dict): Current incident case
- `node_clue_tags` (list): Node clue tags
- `edge_clue_tags` (list): Edge clue tags
- `a` (dict): Initial clue weights
- `get_edge_weight` (function): Edge weight calculation function
- `edge_backward_factor` (float): Backward edge factor
- `historical_incident_topologies` (list): Historical incident data
- `init_clue_tag` (str): Initial clue tag
- `range_a` (float): Optimization range for weights
- `num_trials` (int): Number of optimization trials

**Returns:**
- `node_clue_tags` (list): Updated node clue tags
- `a` (dict): Optimized clue weights

##### `eval(historical_incident_topologies, node_clue_tags, edge_clue_tags, a, get_edge_weight, edge_backward_factor)`
Evaluate diagnosis performance on historical data.

**Parameters:**
- `historical_incident_topologies` (list): Historical incident cases
- `node_clue_tags` (list): Node clue tags
- `edge_clue_tags` (list): Edge clue tags
- `a` (dict): Clue weights
- `get_edge_weight` (function): Edge weight function
- `edge_backward_factor` (float): Backward edge factor

**Returns:**
- `reward` (float): Accuracy reward
- `punishment` (float): Regularization punishment

### Data Structures

#### Topology Format
```python
topology = {
    'nodes': ['node1', 'node2', ...],
    'edge_info': {
        'node1_node2': {
            'metric1': [values...],
            'metric2': [values...]
        }
    },
    'node_info': {
        'node1': {
            'metric1': [values...],
            'metric2': [values...]
        }
    },
    'root_cause': 'node_id'  # for labeled data
}
```

#### PyTorch Geometric Data Format
```python
from torch_geometric.data import Data

data = Data(
    x=node_features,      # Node feature matrix
    edge_index=edge_index, # Edge connectivity
    edge_attr=edge_attr,   # Edge features
    global_x=global_features, # Global features
    y=label,              # Graph label
    batch=batch_vector    # Batch assignment
)
```

### Constants

- `DAMPING = 0.1`: PageRank damping factor for root cause localization

### Example Usage

```python
# Incident Detection
from src.incident_detection import callSpatioDevNet

model = callSpatioDevNet(
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    hidden_dim=20
)
model.fit(train_data)
scores, features = model.predict(test_data)

# Incident Diagnosis
from src.incident_diagnosis.incident_diagnosis import (
    get_weight_from_edge_info,
    root_cause_localization,
    optimize
)

# Diagnose root cause
root_cause = root_cause_localization(
    case=incident_case,
    node_clue_tags=['cpu_usage', 'memory_usage'],
    edge_clue_tags=['response_time', 'error_rate'],
    a={'cpu_usage': 1.0, 'memory_usage': 0.8},
    get_edge_weight=get_weight_from_edge_info
)
```
        




## Troubleshooting

- Ensure you have the correct Python version (3.7+)
- Install PyTorch with appropriate CUDA support if using GPU
- For torch_geometric installation issues, refer to the [official documentation](https://pytorch-geometric.readthedocs.io/)
- Make sure all data files are in the correct `./data` directory


# Citation

If you find this work useful, please cite our paper:

```
@inproceedings{DBLP:conf/kbse/HeCLYCYL22,
  author       = {Zilong He and
                  Pengfei Chen and
                  Yu Luo and
                  Qiuyu Yan and
                  Hongyang Chen and
                  Guangba Yu and
                  Fangyuan Li},
  title        = {Graph based Incident Extraction and Diagnosis in Large-Scale Online
                  Systems},
  booktitle    = {37th {IEEE/ACM} International Conference on Automated Software Engineering,
                  {ASE} 2022, Rochester, MI, USA, October 10-14, 2022},
  pages        = {48:1--48:13},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3551349.3556904},
  doi          = {10.1145/3551349.3556904},
  timestamp    = {Thu, 22 Jun 2023 07:45:51 +0200},
  biburl       = {https://dblp.org/rec/conf/kbse/HeCLYCYL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



