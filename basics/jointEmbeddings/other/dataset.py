from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class TextImageDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize images
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        text = self.texts[idx]
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        return image, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)

