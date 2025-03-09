from clip import load as loadClip, tokenize
from torch import nn

class JointEmbeddingModel(nn.Module):
    def __init__(self, device, hidden_dim=256):
        super().__init__()
        self.model, _ = loadClip("ViT-B/32", device=device)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images, texts):
        ...

        # return img_emb, text_emb
