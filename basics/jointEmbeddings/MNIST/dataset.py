from clip import load as loadClip, tokenize
from torch import FloatTensor
from torch.utils.data import Dataset

digit_to_text = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

class MNISTTextDataset(Dataset):
    def __init__(self, device, dataset) -> None:
        self.device = device
        self.dataset = dataset
        _, self.preprocess = loadClip("ViT-B/32", device=device)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> [FloatTensor, FloatTensor]:
        image, label = self.dataset[idx]
        label = tokenize(digit_to_text[label]).to(self.device)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image, label
