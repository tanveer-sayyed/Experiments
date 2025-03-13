"""
    pip install segment-anything opencv-python torch torchvision
"""

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.utils.data import Dataset, DataLoader
import os

class MultiMaskDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if self.transform:  image = self.transform(image)
        return image

class SAMMultiMaskModel:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cuda"):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def generate_masks(self, image):
        masks = self.mask_generator.generate(image) # SAM expects images in [0, 255] range with RGB channels
        multi_mask = np.zeros((image.shape[0], image.shape[1], len(masks))) # Extract multiple masks as binary channels
        for idx, mask_data in enumerate(masks):
            multi_mask[:, :, idx] = mask_data['segmentation']
        return multi_mask.astype(np.uint8)

def train_model(model, dataloader, num_epochs=10, device='cuda'):
    model.sam.train()  # Set SAM to training mode (if finetuning is required)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images in dataloader:
            images = [img.to(device) for img in images]

            for image in images:
                masks = model.generate_masks(image.permute(1, 2, 0).cpu().numpy())
                # Example: Compute loss with custom logic if training SAM
                # Loss design depends on your chosen strategy (e.g., Dice Loss, Boundary IoU)
                loss = torch.tensor(0.0, requires_grad=True).to(device)
                loss.backward()
                total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    transform = lambda x: torch.tensor(x).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

    train_dataset = MultiMaskDataset(image_dir="data/images", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = SAMMultiMaskModel(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth")
    train_model(model, train_loader, num_epochs=15)

