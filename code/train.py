import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from models import CustomGNN
from data_loader import LRGBDataset, TUDataset
from config import config
from metrics import compute_metrics
from rewiring import *
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------- Train Loop -------------------------
def train(model, device, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        # Changed from out = model(batch) to out = model(batch.x, batch.edge_index, batch.batch)
        # This is to ensure that the model can handle the batch structure correctly.
        out = model(batch)         # out.x shape: [num_nodes, num_classes]
        # pooled_out = global_mean_pool(out.x, batch.batch)  # shape: [batch_size, num_classes]
        # loss = F.cross_entropy(pooled_out, batch.y)
        loss = loss_func(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ------------------------- Validation Loop -------------------------
def validate(model, device, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)
            out = model(batch)
            # preds = out.argmax(dim=1)
            # preds = F.softmax(out, dim=1)
            preds = out
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # Calculate metrics (accuracy for now)
    metrics = compute_metrics(all_labels, all_preds, config['dataset']['name'])
    return metrics

# ------------------------- Test Loop -------------------------
def test(model, device, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            out = model(batch)
            # preds = out.argmax(dim=1)
            # preds = F.softmax(out, dim=1)
            preds = out
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # Calculate metrics (accuracy for now)
    metrics = compute_metrics(all_labels, all_preds, config['dataset']['name'])
    return metrics

# ------------------------- Start the Experiment -------------------------
def start():
    # Load configuration from config.py
    dataset_name = config['dataset']['name']
    model_name = config['model']['name']
    rewiring_name = config['train']['rewiring']

    if rewiring_name == 'LASER':
        rewiring_transform = LaserGlobalTransform()
    else:
        rewiring_transform = None
    
    # Initialize Dataset
    if config['dataset']['dataset_type'] == 'lrgb':
        train_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='train', pre_transform=rewiring_transform)
        val_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='val', pre_transform=rewiring_transform)
        test_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='test', pre_transform=rewiring_transform)
    elif config['dataset']['dataset_type'] == 'tu':
        train_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='train', pre_transform=rewiring_transform)
        val_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='val', pre_transform=rewiring_transform)
        test_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='test', pre_transform=rewiring_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # DataLoader for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    
    # Initialize Model
    model = CustomGNN(dim_in=config['dataset']['dim_in'], dim_edge=config['dataset']['dim_edge'] ,dim_out=config['dataset']['dim_out'], model_cfg=config['model']).to(config['device'])
    model.build_conv_model(model_name)
    
    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    loss_func = torch.nn.CrossEntropyLoss() if config['dataset']['name'] in ['peptides-func'] else torch.nn.MSELoss() if config['dataset']['name'] in ['peptides-struct'] else None
    
    # Initialize best metrics for checkpointing
    best_val_metrics = None
    best_model_state = None
    history = {'train_loss': [], 'val_metrics': []}

    # Training Loop
    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
        
        # Training
        train_loss = train(model, config['device'], train_loader, optimizer, loss_func)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        val_metrics = validate(model, config['device'], val_loader)
        print(f"Validation Metrics: {val_metrics}")

        # Save the best model based on validation performance
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)

        if best_val_metrics is None or val_metrics[config['dataset']['best_metric']] > best_val_metrics[config['dataset']['best_metric']]:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model_state)

    # Test the model
    test_metrics = test(model, config['device'], test_loader)
    print(f"\nTest Metrics: {test_metrics}")

    # ------------------------- Save Checkpoint -------------------------
    # Ensure the directory exists
    os.makedirs(f"./checkpoints/{dataset_name}/", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/{dataset_name}/{rewiring_name}_best_model.pth")

    # ------------------------- Plot Training History -------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Train Loss on the first y-axis
    ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for Validation Accuracy
    ax2 = ax1.twinx()
    ax2.plot([metrics[config['dataset']['best_metric']] for metrics in history['val_metrics']], 
             label=f"Validation Accuracy({config['dataset']['best_metric']})", color='tab:orange')
    ax2.set_ylabel(f'Validation Accuracy ({config["dataset"]["best_metric"]})', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add a title and show the plot
    plt.title(f'Training and Validation History for {dataset_name}')
    fig.tight_layout()
    plt.show()

    return best_val_metrics, test_metrics

if __name__ == "__main__":
    start()
