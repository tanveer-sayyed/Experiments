from torch import arange
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        # Normalize embeddings
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity scores (cosine similarity)
        logits = (image_features @ text_features.T) / self.temperature  # (Batch x Batch)

        # Ground truth labels (diagonal should be max similarity)
        labels = arange(logits.shape[0], device=logits.device)

        # Compute loss
        return self.loss_fn(logits, labels) + self.loss_fn(logits.T, labels)  # Symmetric loss
