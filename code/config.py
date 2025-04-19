import torch

# -------------------- Hyperparameters for Training --------------------
train_hyperparameters = {
    'batch_size': 64,                  # Size of each training batch
    'epochs': 2,                     # Total number of training epochs
    'learning_rate': 0.001,            # Learning rate for optimizer
    'weight_decay': 5e-4,              # Weight decay (L2 regularization)
    'dropout': 0.5,                    # Dropout rate used in GatedGCN layers
    'early_stopping_patience': 10,     # Number of epochs to wait for improvement in validation
}

# -------------------- Dataset-specific Configurations --------------------
dataset_config = {
    'peptides-func': {                # Example of a dataset from the LRGB or TU datasets
        'root': './data',             # Root directory where dataset is stored
        'dataset_type': 'lrgb',       # Dataset type (LRGB or TU)
        'name': 'peptides-func',      # Dataset name
        'num_classes': 2,             # Number of classes for classification
        'num_features': 9,           # Number of features per node (example)
    },
    'PROTEINS': {                     # Example of a TU dataset
        'root': './data',
        'dataset_type': 'tu',         # TU dataset type
        'name': 'PROTEINS',           # Dataset name
        'num_classes': 2,             # Number of classes (binary classification)
        'num_features': 3,            # Feature dimension (example for PROTEINS)
    },
    # Additional datasets can be added here
    'CORA': {
        'root': './data',
        'dataset_type': 'tu',         # Type of dataset (TU)
        'name': 'CORA',               # Dataset name
        'num_classes': 7,             # CORA has 7 classes
        'num_features': 1433,         # CORA has 1433 node features
    },
    'CITATION': {
        'root': './data',
        'dataset_type': 'tu',         # Type of dataset (TU)
        'name': 'CITATION',           # Dataset name
        'num_classes': 3,             # Number of classes (binary classification)
        'num_features': 3703,         # Feature dimension (example for CITATION)
    }
}

# -------------------- GatedGCN Model Config --------------------
gatedgcn_config = {
    'name': 'gatedgcn',           # Layer/model name
    'dim_inner': 128,             # Inner hidden dimension
    'layers_pre_mp': 0,           # Pre-message passing layers
    'layers_mp': 4,               # Message passing layers
    'residual': False,            # Residual connections
    'equivstable_pe': False,      # Equivariant Stable PE (for LapPE if used)
    'dropout': 0.5,               # Dropout in GatedGCN layers
    'head': 'mlp'                 # Head to use after GNN layers
}

# -------------------- Final Config Dictionary --------------------
config = {
    'model': gatedgcn_config,             # Configuration for the model (GatedGCN)
    'train': train_hyperparameters,       # Training hyperparameters (batch size, lr, etc.)
    'dataset': dataset_config['peptides-func'],           # Dataset configuration (paths, num classes, etc.)
}

# -------------------- Device Configuration --------------------
# Automatically set device to GPU (CUDA) if available, otherwise use CPU.
device = 'cpu' if torch.cuda.is_available() else 'cpu'

# Example of using the config (for printing or other uses)
print(f"Using device: {device}")

config['device'] = device  # Add device to the config dictionary for later use

