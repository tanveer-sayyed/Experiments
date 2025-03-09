image_paths = ["image1.jpg", "image2.jpg"]
texts = ["A cat sitting on a windowsill", "A dog running in the park"]

# Create dataset and dataloader
dataset = TextImageDataset(image_paths, texts, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, optimizer, and loss
model = JointEmbeddingModel().cuda()
loss_fn = ContrastiveLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    for images, input_ids, attention_mask in dataloader:
        images, input_ids, attention_mask = images.cuda(), input_ids.cuda(), attention_mask.cuda()

        # Forward pass
        img_emb, text_emb = model(images, input_ids, attention_mask)
        loss = loss_fn(img_emb, text_emb)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
