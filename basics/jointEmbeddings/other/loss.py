class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute similarity matrix
        logits = (image_features @ text_features.T) / self.temperature  # Cosine similarity

        # Ground truth labels (diagonal matrix)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Compute loss
        return self.loss_fn(logits, labels) + self.loss_fn(logits.T, labels)
