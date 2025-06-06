import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

def get_outlier_scores(model, data_loader, sph_center, gamma, device):
    """
    Computes outlier scores for each sample in the given data loader using the provided model
    and sphere center. The outlier score is the Euclidean distance between the model's output
    for a given input and the sphere center in the latent space.

    :param model: The PyTorch model used for feature extraction.
    :param data_loader: DataLoader instance supplying the input samples.
    :param sph_center: A tensor representing the sphere center in the embedding space.
    :param gamma: The hyperparameter gamma used in the DASVDD loss.
    :param device: The torch device (e.g., 'cuda' or 'cpu') where the computation is performed.
    :return: A numpy array containing the outlier scores for all samples in the data loader.
    """
    scores_list = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            x_hat = model(inputs)
            phi_x = model.encoder(inputs)
            rec_error = torch.sum((x_hat - inputs) ** 2, dim=1).sqrt()
            dist = torch.sum((phi_x - sph_center) ** 2, dim=1).sqrt()
            score = rec_error + gamma * dist
            scores_list.append(score.cpu().numpy())

    return np.concatenate(scores_list)


def eval_d_svdd(scores, true_labels, percentile=95):
    """
    Evaluate the metrics for a Deep SVDD model based on the provided anomaly scores, true
    labels, and a specified percentile to determine the threshold. Metrics such as AUC-ROC,
    AUC-PR, F1-Score, and Recall are computed to assess the performance.

    :param scores:
        A NumPy array or list representing the anomaly scores for each sample. Higher
        scores typically indicate higher likelihood of being an anomaly.

    :param true_labels:
        A NumPy array or list representing the ground truth labels. These labels should be
        binary, where 1 indicates an anomaly and 0 indicates normal.

    :param percentile:
        An integer (default: 95) specifying the percentile value used to compute the
        threshold. Scores above this threshold are classified as anomalies.

    :return:
        A dictionary containing the evaluation metrics:
            - 'AUC-ROC': Area Under the Receiver Operating Characteristic Curve.
            - 'AUC-PR': Area Under the Precision-Recall Curve.
            - 'F1-Score': The F1 Score, which is the harmonic mean of precision and recall.
            - 'Recall': The fraction of correctly identified anomalies among all true
              anomalies.
    """

    # Convert the score to predictions using a threshold
    threshold = np.percentile(scores, percentile)
    predictions = (scores >= threshold).astype(int)

    # Compute evaluation metrics
    metrics = {
        'AUC-ROC': roc_auc_score(true_labels, scores),
        'AUC-PR': average_precision_score(true_labels, scores),
        'F1-Score': f1_score(true_labels, predictions),
        'Recall': recall_score(true_labels, predictions)
    }

    return metrics