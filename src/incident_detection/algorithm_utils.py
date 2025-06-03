"""
Algorithm Utilities Module

This module contains base classes and utilities for implementing machine learning
algorithms. It provides abstract base
classes and PyTorch utilities for consistent algorithm implementation.
"""

import abc
import logging
import random

import numpy as np
import torch
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    """
    Abstract base class for algorithms.
    
    This class provides a common interface and basic functionality that all
    algorithm implementations should inherit from. It handles logging, seeding,
    and maintains prediction details.
    
    Attributes:
        logger: Logger instance for the algorithm
        name: Human-readable name of the algorithm
        seed: Random seed for reproducibility
        details: Flag to enable detailed logging/tracking
        prediction_details: Dictionary to store prediction metadata
    """
    
    def __init__(self, module_name, name, seed, details=False):
        """
        Initialize the Algorithm base class.
        
        Args:
            module_name (str): Name of the module for logging purposes
            name (str): Human-readable name of the algorithm
            seed (int, optional): Random seed for reproducibility
            details (bool): Whether to enable detailed tracking
        """
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}

        # Set random seeds for reproducibility if provided
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        """
        Return string representation of the algorithm.
        
        Returns:
            str: The algorithm name
        """
        return self.name

    @abc.abstractmethod
    def fit(self, datalist):
        """
        Train the algorithm on the given dataset.
        
        This method must be implemented by all subclasses to define
        how the algorithm learns from training data.
        
        Args:
            datalist: Training data in the format expected by the algorithm
        """
        pass

    @abc.abstractmethod
    def predict(self, datalist):
        """
        Generate predictions for the given data.
        
        This method must be implemented by all subclasses to define
        how the algorithm detection results.
        
        Args:
            datalist: Data to generate predictions for
            
        Returns:
            Detection scores for the input data
        """
        pass


class PyTorchUtils(metaclass=abc.ABCMeta):
    """
    Utility class providing common PyTorch functionality.
    
    This class handles device management, variable creation, and model
    placement for PyTorch-based algorithms. It ensures consistent
    GPU/CPU usage and random seed management.
    
    Attributes:
        gpu: GPU device ID to use (None for CPU)
        seed: Random seed for PyTorch operations
        framework: Framework identifier (0 for PyTorch)
    """
    
    def __init__(self, seed, gpu):
        """
        Initialize PyTorch utilities.
        
        Args:
            seed (int, optional): Random seed for PyTorch operations
            gpu (int, optional): GPU device ID (None for CPU)
        """
        self.gpu = gpu
        self.seed = seed
        
        # Set PyTorch random seeds for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            
        self.framework = 0  # Framework identifier

    @property
    def device(self):
        """
        Get the appropriate PyTorch device (GPU or CPU).
        
        Returns:
            torch.device: CUDA device if available and specified, otherwise CPU
        """
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        """
        Convert tensor to Variable and move to appropriate device.
        
        Args:
            t (torch.Tensor): Input tensor
            **kwargs: Additional arguments for Variable creation
            
        Returns:
            torch.autograd.Variable: Variable on the appropriate device
        """
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        """
        Move a PyTorch model to the appropriate device.
        
        Args:
            model (torch.nn.Module): PyTorch model to move
        """
        model.to(self.device)
