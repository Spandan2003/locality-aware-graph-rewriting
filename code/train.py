import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from models import CustomGNN
from data_loader import LRGBDataset, TUDataset
from config import config
from metrics import compute_metrics
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ------------------------- Train Loop -------------------------
def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out.x, batch.y)
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
            preds = out.x.argmax(dim=1)
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
            preds = out.x.argmax(dim=1)
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
    
    # Initialize Dataset
    if config['dataset']['dataset_type'] == 'lrgb':
        train_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='train')
        val_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='val')
        test_dataset = LRGBDataset(root=config['dataset']['root'], name=config['dataset']['name'], split='test')
    elif config['dataset']['dataset_type'] == 'tu':
        train_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='train')
        val_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='val')
        test_dataset = TUDataset(root=config['dataset']['root'], name=dataset_name, split='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # DataLoader for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    
    # Initialize Model
    model = CustomGNN(dim_in=train_dataset.num_features, dim_out=train_dataset.num_classes, model_cfg=config['model']).to(config['device'])
    model.build_conv_model(model_name)
    
    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    
    # Initialize best metrics for checkpointing
    best_val_metrics = None
    best_model_state = None
    history = {'train_loss': [], 'val_metrics': []}

    # Training Loop
    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
        
        # Training
        train_loss = train(model, config['device'], train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        val_metrics = validate(model, config['device'], val_loader)
        print(f"Validation Metrics: {val_metrics}")

        # Save the best model based on validation performance
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)

        if best_val_metrics is None or val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model_state)

    # Test the model
    test_metrics = test(model, config['device'], test_loader)
    print(f"\nTest Metrics: {test_metrics}")

    # ------------------------- Save Checkpoint -------------------------
    torch.save(model.state_dict(), f"best_model_{dataset_name}.pth")

    # ------------------------- Plot Training History -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot([metrics['accuracy'] for metrics in history['val_metrics']], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(f'Training and Validation History for {dataset_name}')
    plt.show()

    return best_val_metrics, test_metrics

if __name__ == "__main__":
    start()
    