"""
Incident Detection Module of Gem

This module implements the Gem algorithm for incident detection using
issue impact topologies. It uses Graph Neural Networks (GNNs) to analyze spatial
and temporal patterns in system metrics to detect incidents.

The implementation includes:
- Main algorithm class 
- Neural network module with GNN layers
- Training and prediction functionality
- Cold start prediction 
"""

import logging

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from tqdm import trange,tqdm
from torch_geometric.nn import GATConv, NNConv, global_max_pool, global_mean_pool, global_add_pool, GlobalAttention
import IPython
import matplotlib.pyplot as plt
from torch.nn.init import xavier_normal_, zeros_
from torch_geometric.data import Data, DataLoader
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from river import neighbors

from .algorithm_utils import Algorithm, PyTorchUtils

def bf_search(labels, scores):
    """
    Brute force search to find the optimal threshold for binary classification.
    
    This function searches through all possible threshold values to find the one
    that maximizes the F1-score on the training set. This threshold can then be
    used for making binary predictions.
    
    Args:
        labels (array-like): True binary labels (0 or 1)
        scores (array-like): Predicted anomaly scores (continuous values)
    
    Returns:
        tuple: A tuple containing:
            - list: Precision, recall, F1-score, and support for best threshold
            - float: The optimal threshold value that maximizes F1-score
    """
    # Initialize best metrics and threshold
    m = (-1., -1., -1., None)  # (precision, recall, f1, support)
    m_t = 0.0  # best threshold
    
    # Try all unique score values as potential thresholds (excluding extremes)
    for threshold in sorted(list(scores))[1:-1]:
        # Convert scores to binary predictions using current threshold
        predictions = (scores > threshold).astype('int')
        
        # Calculate precision, recall, F1-score for current threshold
        target = precision_recall_fscore_support(labels, predictions, average='binary')
        
        # Update best metrics if current F1-score is better
        if target[2] > m[2]:  # target[2] is F1-score
            m_t = threshold
            m = target
    
    print(m, m_t)  # Print best metrics and threshold
    return m, m_t


class callSpatioDevNet(Algorithm, PyTorchUtils):
    """
    Gem Algorithm for Incident Detection.
    
    This class implements the Gem algorithm, which uses Graph Neural Networks
    to detect incidents using issue impact topologies. It combines spatial information from
    issue impact topology with temporal patterns in metrics to identify anomalies.
    
    The algorithm uses:
    - Graph Attention Networks (GAT) for spatial feature learning
    - Neural Network Convolution (NNConv) for edge feature processing
    - Global pooling for graph-level representations
    - Multiple loss functions for different training scenarios
    
    Attributes:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate for optimization
        input_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden representations
        edge_attr_len (int): Length of edge attribute vectors
        global_fea_len (int): Length of global feature vectors
        num_layers (int): Number of GNN layers
        devnet (SpatioDevNetModule): The neural network module
        loss_logs (dict): Training loss and metrics history
    """
    
    def __init__(self, name: str = 'SpatioDevNetPackage', num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
                 input_dim: int = None, hidden_dim: int = 20, edge_attr_len: int = 60, global_fea_len: int = 2,
                 num_layers: int = 2, edge_module: str = 'linear', act: bool = True, pooling: str = 'attention', 
                 is_bilinear: bool = False, nonlinear_scorer: bool = False, head: int = 4, aggr: str = 'mean', 
                 concat: bool = False, dropout: float = 0.4, weight_decay: float = 1e-2, loss_func: str = 'focal_loss', 
                 seed: int = None, gpu: int = None, ipython=True, details=True):
        """
        Initialize the SpatioDevNet algorithm of Gem.
        
        Args:
            name (str): Name identifier for the algorithm instance
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lr (float): Learning rate for Adam optimizer
            input_dim (int): Dimension of input node features
            hidden_dim (int): Dimension of hidden GNN representations
            edge_attr_len (int): Length of edge attribute vectors
            global_fea_len (int): Length of global feature vectors
            num_layers (int): Number of GNN layers
            edge_module (str): Type of edge processing module ('linear' or 'lstm')
            act (bool): Whether to use activation functions
            pooling (str): Global pooling method ('attention', 'max', 'mean', 'add')
            is_bilinear (bool): Whether to use bilinear final scorer
            nonlinear_scorer (bool): Whether to use nonlinear final scorer
            head (int): Number of attention heads for GAT
            aggr (str): Aggregation method for GNN ('mean', 'max', 'add')
            concat (bool): Whether to concatenate attention heads
            dropout (float): Dropout probability
            weight_decay (float): L2 regularization weight
            loss_func (str): Loss function type ('focal_loss', 'dev_loss', 'cross_entropy')
            seed (int): Random seed for reproducibility
            gpu (int): GPU device ID (None for CPU)
            ipython (bool): Whether running in IPython environment
            details (bool): Whether to enable detailed logging
        """
        # Initialize parent classes
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        
        # Training hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Model architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_len = edge_attr_len
        self.global_fea_len = global_fea_len
        self.num_layers = num_layers

        # Model configuration
        self.edge_module = edge_module
        self.act = act
        self.pooling = pooling
        self.is_bilinear = is_bilinear
        self.nonlinear_scorer = nonlinear_scorer
        self.head = head
        self.aggr = aggr
        self.concat = concat
        self.dropout = dropout
        self.weight_decay = weight_decay

        # Training state
        self.final_train_fscore = None
        self.ipython = ipython
        self.loss_func = loss_func

        # Initialize the neural network module
        self.devnet = SpatioDevNetModule(
            self.input_dim, self.hidden_dim, self.edge_attr_len, self.global_fea_len, 
            self.num_layers, self.edge_module, self.act, self.pooling, self.is_bilinear, 
            self.nonlinear_scorer, self.head, self.aggr, self.concat, self.dropout, 
            self.seed, self.gpu
        )
        
        # Training logs for monitoring
        self.loss_logs = {}

    def fit(self, datalist: list, valid_list: list = None, log_step: int = 20, patience: int = 10, 
            valid_proportion: float = 0.0, early_stop_fscore: float = None):
        """
        Train the SpatioDevNet model of Gem on the provided dataset.
        
        This method implements the complete training loop with support for validation,
        early stopping, and comprehensive logging. It handles data loading, loss
        computation, optimization, and model checkpointing.
        
        Args:
            datalist (list): List of PyTorch Geometric Data objects for training
            valid_list (list, optional): Separate validation dataset
            log_step (int): Frequency of logging and plotting (every N iterations)
            patience (int): Number of epochs to wait before early stopping
            valid_proportion (float): Proportion of training data to use for validation
            early_stop_fscore (float, optional): F1-score threshold for early stopping
        """
        # Setup validation strategy
        if valid_list is not None:
            # Use provided validation set
            train_list = datalist
            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)
            valid_loader = DataLoader(dataset=valid_list, batch_size=len(valid_list), shuffle=False)
        elif valid_proportion != 0:
            # Split training data for validation
            split_point = int(valid_proportion * len(datalist))
            shuffle_list = copy.deepcopy(datalist)
            random.shuffle(shuffle_list)

            train_list = shuffle_list[:-split_point]
            valid_list = shuffle_list[-split_point:]

            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)
            valid_loader = DataLoader(dataset=valid_list, batch_size=len(valid_list), shuffle=False)
        else:
            # No validation - use all data for training
            train_list = datalist
            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)

        # Move model to appropriate device and setup optimizer
        self.to_device(self.devnet)
        optimizer = torch.optim.Adam(self.devnet.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training state variables
        iters_per_epoch = len(train_loader)
        counter = 0  # Early stopping counter
        best_val_fscore = 0
        best_train_fscore = 0

        # Main training loop
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            self.devnet.train()  # Set model to training mode
            
            # Iterate through training batches
            for (i, batch) in enumerate(tqdm(train_loader)):
                # Forward pass
                output, feature = self.devnet(batch)
                
                # Compute loss based on specified loss function
                if self.loss_func == 'focal_loss':
                    total_loss = SpatioDevNetModule.bce_focal_loss_function(output, batch.y)
                elif self.loss_func == 'dev_loss':
                    total_loss = SpatioDevNetModule.deviation_loss_function(output, batch.y, batch.confidence)
                else:
                    total_loss = SpatioDevNetModule.cross_entropy_loss_function(output, batch.y)

                # Store loss for logging
                loss = {'total_loss': total_loss.data.item()}

                # Backward pass and optimization
                self.devnet.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Periodic logging and visualization
                if (i+1) % log_step == 0:
                    self._log_and_plot_progress(loss, epoch, i, iters_per_epoch)
            
            # End-of-epoch validation and early stopping logic
            if valid_proportion != 0 or valid_list is not None:
                self._validate_and_check_early_stopping(
                    train_list, valid_list, valid_loader_of_train_data, 
                    valid_loader, counter, best_val_fscore, patience
                )
            elif early_stop_fscore is not None:
                self._check_train_early_stopping(
                    train_list, valid_loader_of_train_data, counter, 
                    best_train_fscore, patience, early_stop_fscore
                )
        
        # Save final model
        torch.save(self.devnet.state_dict(), self.name+'.pt')

    def _log_and_plot_progress(self, loss, epoch, iteration, iters_per_epoch):
        """
        Log training progress and create visualization plots.
        
        Args:
            loss (dict): Current loss values
            epoch (int): Current epoch number
            iteration (int): Current iteration within epoch
            iters_per_epoch (int): Total iterations per epoch
        """
        # Clear output for IPython environments
        if self.ipython:
            IPython.display.clear_output()
            plt.figure(figsize=(12, 6))
        else:
            plt.figure(figsize=(12, 6))
        
        # Create log message
        log = "Epoch [{}/{}], Iter [{}/{}]".format(
            epoch+1, self.num_epochs, iteration+1, iters_per_epoch)
        
        for tag, value in loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        # Update loss logs and create plots
        plt_ctr = 1
        if not self.loss_logs:
            # Initialize loss logs
            for loss_key in loss:
                self.loss_logs[loss_key] = [loss[loss_key]]
                plt.subplot(2,3,plt_ctr)
                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                plt.legend()
                plt_ctr += 1
        else:
            # Update existing loss logs
            for loss_key in loss:
                self.loss_logs[loss_key].append(loss[loss_key])
                plt.subplot(2,3,plt_ctr)
                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                plt.legend()
                plt_ctr += 1
            
            # Plot training metrics if available
            if 'train_prf' in self.loss_logs:
                plt.subplot(2, 3, plt_ctr)
                plt.plot(np.array(self.loss_logs['train_prf'])[:, 0], label='train_precision')
                plt.plot(np.array(self.loss_logs['train_prf'])[:, 1], label='train_recall')
                plt.plot(np.array(self.loss_logs['train_prf'])[:, 2], label='train_fscore')
                plt.legend()
                plt_ctr += 1
            
            # Plot validation metrics if available
            if 'valid_loss' in self.loss_logs:
                for valid_item in ['valid_loss', 'valid_precision', 'valid_recall', 'valid_fscore']:
                    plt.subplot(2,3,plt_ctr)
                    plt.plot(np.array(self.loss_logs[valid_item]), label=valid_item)
                    plt.legend()
                    plt_ctr += 1
                print("valid_fscore:", self.loss_logs['valid_fscore'])
        
        # Display or save plot
        if self.ipython:
            plt.show()
        else:
            plt.savefig("test.png", dpi=120)

    def load(self, model_file: str = None):
        """
        Load a pre-trained model from file.
        
        Args:
            model_file (str, optional): Path to model file. If None, uses default name.
        """
        if model_file is None:
            self.devnet.load_state_dict(torch.load(self.name+'.pt'))
        else:
            self.devnet.load_state_dict(torch.load(model_file))

    def predict(self, datalist: list):
        """
        Generate predictions for the given dataset.
        
        Args:
            datalist (list): List of PyTorch Geometric Data objects
            
        Returns:
            tuple: (anomaly_scores, feature_representations)
                - anomaly_scores (np.ndarray): Predicted anomaly scores
                - feature_representations (np.ndarray): Learned feature representations
        """
        data_loader = DataLoader(dataset=datalist, batch_size=self.batch_size, shuffle=False)
        self.devnet.eval()  # Set model to evaluation mode

        outputs = []
        features = []

        # Generate predictions without gradient computation
        with torch.no_grad():
            for (i, batch) in enumerate(tqdm(data_loader)):
                output, feature = self.devnet(batch)
                outputs.append(output.data.cpu().numpy())
                features.append(feature.data.cpu().numpy())

        # Concatenate all batch results
        outputs = np.concatenate(outputs)
        features = np.concatenate(features)

        return outputs, features
    
    def cold_start_predict(self, datalist: list, n_neighbors: int = 3):
        """
        Perform cold start prediction with an online retrieval based knn model.

        Args:
            datalist (list): List of PyTorch Geometric Data objects
            n_neighbors (int): Number of neighbors for k-NN classification
            
        Returns:
            tuple: (predictions, prediction_probabilities, knn_model)
                - predictions (list): Binary predictions (0 or 1)
                - prediction_probabilities (list): Prediction confidence scores
                - knn_model: Trained k-NN model for future use
        """
        # Get feature representations from the neural network
        _, feas = self.predict(datalist)
        labels = [int(item.y) for item in datalist]

        # Initialize k-NN classifier and prediction lists
        knn_preds = []
        knn_pred_proba = []
        knn = neighbors.KNNClassifier(window_size=5000, n_neighbors=n_neighbors)

        # Handle first sample (no neighbors available yet)
        knn_preds.append(1)  # Default prediction
        knn_pred_proba.append(1.0)  # Default confidence
        fea_dict = {i: x for i, x in enumerate(feas[0])}
        knn.learn_one(fea_dict, labels[0])
        
        # Process remaining samples
        for fea, label in zip(feas[1:], labels[1:]):
            fea_dict = {i: x for i, x in enumerate(fea)}
            pred = knn.predict_one(fea_dict)
            pred_proba = knn.predict_proba_one(fea_dict).get(1, 0)
            knn_preds.append(pred)
            knn_pred_proba.append(pred_proba)
            
            # Update k-NN model with current sample
            knn.learn_one(fea_dict, label)

        return knn_preds, knn_pred_proba, knn


class SpatioDevNetModule(nn.Module, PyTorchUtils):
    """
    Neural Network Module for SpatioDevNet.
    
    This module implements the core neural network architecture combining:
    - NNConv layers for processing edge features and node features
    - GAT (Graph Attention) layers for learning spatial relationships
    - Global pooling for graph-level representations
    - Final scoring layers for anomaly detection
    
    The architecture is designed to capture both local spatial patterns
    and global graph-level characteristics for effective anomaly detection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 4, edge_attr_len: int = 60, 
                 global_fea_len: int = 2, num_layers: int = 2, edge_module: str = 'linear', 
                 act: bool = True, pooling: str = 'attention', is_bilinear: bool = False, 
                 nonlinear_scorer: bool = False, head: int = 4, aggr: str = 'mean', 
                 concat: bool = False, dropout: float = 0.5, seed: int = 0, gpu: int = None):
        """
        Initialize the SpatioDevNet neural network module.
        
        Args:
            input_dim (int): Dimension of input node features
            hidden_dim (int): Dimension of hidden representations
            edge_attr_len (int): Length of edge attribute vectors
            global_fea_len (int): Length of global feature vectors
            num_layers (int): Number of GNN layers
            edge_module (str): Type of edge processing ('linear' or 'lstm')
            act (bool): Whether to use activation functions
            pooling (str): Global pooling method
            is_bilinear (bool): Whether to use bilinear final scorer
            nonlinear_scorer (bool): Whether to use nonlinear final scorer
            head (int): Number of attention heads
            aggr (str): Aggregation method for message passing
            concat (bool): Whether to concatenate attention heads
            dropout (float): Dropout probability
            seed (int): Random seed
            gpu (int): GPU device ID
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_len = edge_attr_len
        self.global_fea_len = global_fea_len
        self.num_layers = num_layers
        self.edge_module = edge_module
        self.act = act
        self.pooling = pooling
        self.is_bilinear = is_bilinear
        self.nonlinear_scorer = nonlinear_scorer
        self.head = head
        self.aggr = aggr
        self.concat = concat
        self.dropout = dropout

        # Initialize edge processing layer
        if self.edge_module == 'lstm':
            # Use LSTM for temporal edge feature processing
            self.intermediate = NNConv(
                self.input_dim, self.hidden_dim,
                LSTMhelper(self.edge_attr_len, self.input_dim * self.hidden_dim), 
                self.aggr
            )
        else:
            # Use linear transformation for edge features
            self.intermediate = NNConv(
                self.input_dim, self.hidden_dim,
                nn.Linear(self.edge_attr_len, self.input_dim * self.hidden_dim), 
                self.aggr
            )
        self.to_device(self.intermediate)

        # Graph Attention layer for spatial feature learning
        self.local_scorer = GATConv(
            self.hidden_dim, self.global_fea_len, self.head, 
            self.concat, dropout=self.dropout
        )
        self.to_device(self.local_scorer)

        # Global pooling layer
        if self.pooling == 'attention':
            self.attention_pooling = GlobalAttention(nn.Linear(self.global_fea_len, 1))
            self.to_device(self.attention_pooling)
        else:
            self.attention_pooling = None

        # Final scoring layers
        if self.nonlinear_scorer:
            # Nonlinear scorer with residual connection
            self.final_scorer_res = nn.Sequential(
                nn.Linear(2 * self.global_fea_len, 4 * self.global_fea_len),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=self.dropout),
                nn.Linear(4 * self.global_fea_len, self.global_fea_len)
            )
            self.final_scorer = nn.Linear(self.global_fea_len, 1)
        else:
            # Linear or bilinear scorer
            if self.is_bilinear:
                self.final_scorer = nn.Bilinear(self.global_fea_len, self.global_fea_len, 1)
            else:
                self.final_scorer = nn.Linear(2 * self.global_fea_len, 1)
        self.to_device(self.final_scorer)

    def forward(self, data):
        """
        Forward pass through the SpatioDevNet architecture.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_attr: Edge features
                - global_x: Global graph features
                - batch: Batch assignment for nodes
                
        Returns:
            tuple: (anomaly_score, feature_representation)
                - anomaly_score: Predicted anomaly score for the graph
                - feature_representation: Learned feature vector
        """
        # Process node and edge features through NNConv layer
        representation = self.intermediate(data.x, data.edge_index, data.edge_attr)

        # Apply activation and dropout
        if self.act:
            representation = F.relu(representation)
        representation = F.dropout(representation, p=self.dropout, training=self.training)

        # Apply Graph Attention for spatial feature learning
        scores = self.local_scorer(representation, data.edge_index)

        # Global pooling to get graph-level representation
        if self.pooling == 'max':
            local_score_summary = global_max_pool(scores, data.batch)
        elif self.pooling == 'mean':
            local_score_summary = global_mean_pool(scores, data.batch)
        elif self.pooling == 'add':
            local_score_summary = global_add_pool(scores, data.batch)
        else:  # attention pooling
            local_score_summary = self.attention_pooling(scores, data.batch)
        
        # Final scoring
        if self.nonlinear_scorer:
            # Nonlinear scorer with residual connection
            res = self.final_scorer_res(torch.cat((local_score_summary, data.global_x), 1))
            final_repr = local_score_summary + res
            return self.final_scorer(final_repr), final_repr
        else:
            # Linear or bilinear scorer
            if self.is_bilinear:
                return (self.final_scorer(local_score_summary, data.global_x), 
                       torch.cat((local_score_summary, data.global_x), 1))
            else:
                combined_features = torch.cat((local_score_summary, data.global_x), 1)
                return self.final_scorer(combined_features), combined_features

    @staticmethod
    def deviation_loss_function(preds, labels, confidence_margin):
        """
        Deviation-based loss function for incident detection.
        
        This loss function encourages normal samples to have scores close to
        a reference distribution, while anomalous samples should deviate significantly.
        
        Args:
            preds (torch.Tensor): Predicted incident scores
            labels (torch.Tensor): True labels (0 for normal, 1 for incident)
            confidence_margin (torch.Tensor): Confidence margins for each sample
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Generate reference normal distribution
        ref = torch.normal(mean=0., std=1.0, size=(5000,))
        
        # Compute deviation from reference distribution
        dev = (preds - torch.mean(ref)) / torch.std(ref)
        dev = dev.squeeze()
        
        # Loss components
        inlier_loss = torch.abs(dev)  # Normal samples should have low deviation
        outlier_loss = torch.abs(torch.max(confidence_margin - dev, torch.zeros_like(confidence_margin)))
        
        # Weighted combination based on labels
        return torch.mean((1 - labels) * inlier_loss + labels * outlier_loss)

    @staticmethod
    def cross_entropy_loss_function(preds, labels):
        """
        Standard binary cross-entropy loss.
        
        Args:
            preds (torch.Tensor): Raw predictions (logits)
            labels (torch.Tensor): True binary labels
            
        Returns:
            torch.Tensor: BCE loss value
        """
        sig = nn.Sigmoid()
        loss = nn.BCELoss()
        return loss(sig(preds), labels)

    @staticmethod
    def bce_focal_loss_function(preds, labels, alpha=0.5, gamma=0.5):
        """
        Focal loss for handling class imbalance in anomaly detection.
        
        Focal loss down-weights easy examples and focuses learning on hard examples,
        which is particularly useful for imbalanced anomaly detection datasets.
        
        Args:
            preds (torch.Tensor): Raw predictions (logits)
            labels (torch.Tensor): True binary labels
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
            
        Returns:
            torch.Tensor: Focal loss value
        """
        pt = torch.sigmoid(preds)
        
        # Focal loss formula
        loss = - 2 * alpha * (1 - pt) ** gamma * labels * torch.log(pt) - \
               2 * (1 - alpha) * pt ** gamma * (1 - labels) * torch.log(1 - pt)
        
        return torch.mean(loss)


class LSTMhelper(nn.Module, PyTorchUtils):
    """
    LSTM helper module for processing temporal edge features.
    
    This module provides LSTM-based processing of edge attributes,
    which is useful when edge features have temporal dependencies.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 seed: int = 0, gpu: int = None):
        """
        Initialize LSTM helper.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden state dimension
            num_layers (int): Number of LSTM layers
            seed (int): Random seed
            gpu (int): GPU device ID
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

    def forward(self, data):
        """
        Process data through LSTM and return final hidden state.
        
        Args:
            data (torch.Tensor): Input sequence data
            
        Returns:
            torch.Tensor: Final hidden state from LSTM
        """
        return self.lstm(data)[1][0][-1]
