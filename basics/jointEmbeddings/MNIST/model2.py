import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer

class JointEmbeddingModel(nn.Module):
    def __init__(self, img_embedding_dim=512, text_embedding_dim=512, hidden_dim=256):
        super().__init__()
        # Vision Encoder: Pretrained ResNet (or ViT)
        resnet = models.resnet50(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.vision_proj = nn.Linear(resnet.fc.in_features, hidden_dim)
        # Text Encoder: Pretrained BERT (or TinyBERT for speed)
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        # Normalization layers
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images, input_ids, attention_mask):
        # Encode images
        img_emb = self.vision_encoder(images).squeeze(-1).squeeze(-1)
        img_emb = self.vision_proj(img_emb)
        img_emb = self.image_norm(img_emb)

        # Encode text
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_emb = self.text_proj(text_emb)
        text_emb = self.text_norm(text_emb)

        return img_emb, text_emb
