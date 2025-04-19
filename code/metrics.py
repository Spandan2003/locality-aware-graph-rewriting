from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error
import numpy as np

def compute_metrics(y_true, y_pred, dataset_name):
    """
    Computes classification metrics based on the dataset.
    
    Args:
    - y_true: List or numpy array of true labels.
    - y_pred: List or numpy array of predicted labels.
    - dataset_name: The name of the dataset to determine the metric type.
    
    Returns:
    - A dictionary with the computed metrics.
    """
    if dataset_name == "peptides-func":
        # Average Precision (AP) for peptides-func dataset
        avg_precision = average_precision_score(y_true, y_pred)
        return {
            'average_precision': avg_precision
        }

    elif dataset_name == "stuct":
        # Mean Absolute Error (MAE) for -stuct dataset
        mae = mean_absolute_error(y_true, y_pred)
        return {
            'mae': mae
        }

    elif dataset_name == "pcqm-contact":
        # Mean Reciprocal Rank (MRR) for pcqm-contact dataset
        ranks = np.argsort(y_pred, axis=1)  # Sort predictions
        mrr = np.mean(1 / (ranks + 1))  # MRR is the reciprocal of the rank
        return {
            'mrr': mrr
        }

    else:
        # Default to accuracy for all TUDatasets
        accuracy = accuracy_score(y_true, y_pred)
        return {
            'accuracy': accuracy
        }