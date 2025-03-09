import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

from dataset import MNISTTextDataset
from model import JointEmbeddingModel
from loss import ContrastiveLoss

EPOCHS = 100
BATCH_SIZE = 4096

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize like in standard MNIST training
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=False)
train_data = MNISTTextDataset(train_dataset)
test_data = MNISTTextDataset(test_dataset)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointEmbeddingModel(embedding_dim=256).to(device)
loss_fn = ContrastiveLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e+1)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, text_labels, labels in tqdm(train_loader):
        images = images.to(device)
        text_labels = labels.to(device)
        img_emb, text_emb = model(images, text_labels)
        loss = loss_fn(img_emb, text_emb)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

model.eval() # here outside training loop
with torch.no_grad():
    correct = 0
    total = 0
    for images, text_labels, labels in test_loader:
        images = images.to(device)
        text_labels = labels.to(device)
        img_emb, text_emb = model(images, text_labels)
        similarities = (img_emb @ text_emb.T)
        predictions = similarities.argmax(dim=1)
        correct += (predictions == torch.arange(len(labels), device=device)).sum().item()
        total += labels.size(0)
    print(f"Test Accuracy (Alignment Check): {100 * correct / total:.2f}%")
