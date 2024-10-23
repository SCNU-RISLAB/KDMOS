import torch
import torch.nn.functional as F

def lovasz_softmax(probas, labels, classes='present'):
    """Computes the Lovasz-Softmax loss.
    
    Args:
        probas (torch.Tensor): Predictions (N, C) after softmax.
        labels (torch.Tensor): Ground truth labels (N).
        classes (list or str): Classes to compute the loss for.
        
    Returns:
        torch.Tensor: Computed Lovasz-Softmax loss.
    """
    # Get the number of classes from the predictions shape
    num_classes = probas.size(1)
    
    if classes == 'present':
        classes = range(num_classes)

    loss = 0.0
    for c in classes:
        # Get the probabilities and labels for the current class
        fg = (labels == c).float()  # Foreground mask
        if fg.sum() == 0:
            continue  # Skip if there are no true instances of this class

        # Compute the Lovasz loss
        # Sort the predictions and labels
        probas_c = probas[:, c]
        fg = fg[fg != 0]
        probas_c = probas_c[fg.bool()]
        
        # Calculate the Lovasz loss for class c
        loss += lovasz_loss(probas_c, fg)

    return loss

def lovasz_loss(probas, labels):
    """Computes the Lovasz loss."""
    # Sort by probability
    perm = torch.sort(probas, descending=True)[1]
    labels = labels[perm]

    # Compute intersection and union
    intersection = torch.cumsum(labels, dim=0)
    union = torch.arange(1, len(labels) + 1, device=labels.device).float() - intersection + intersection[-1]

    # Compute the Lovasz loss
    loss = 1 - intersection / union
    loss = torch.mean(loss)  # Average over all pixels

    return loss
