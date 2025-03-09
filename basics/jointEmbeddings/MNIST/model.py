import torch.nn as nn
from torchvision import models


class JointEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, embedding_dim)
        )

        self.vocab_size = 10  # 0 to 9
        self.embedding_dim = embedding_dim
        self.text_embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # Normalization layers
        self.text_norm =  nn.LayerNorm(embedding_dim)
        self.image_norm = nn.LayerNorm(embedding_dim)

    def forward(self, images, text_labels):
        img_emb = self.image_norm(
            self.image_encoder(images)
        )
        text_emb = self.text_norm(
             self.text_embedding(text_labels)
        )
        return img_emb, text_emb
