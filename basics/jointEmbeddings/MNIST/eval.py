import matplotlib.pyplot as plt
from dataset import digit_to_text

def infer_text_to_image(text_label):
    """
    Given a text label (e.g., 'seven'), find the most similar image in the test set.
    """
    text_tensor = torch.tensor([digit_to_text.index(text_label)], device=device)

    with torch.no_grad():
        text_embedding = model.text_embedding(text_tensor)
        text_embedding = model.text_norm(text_embedding)

        best_match_idx = -1
        best_similarity = -float("inf")
        best_match_image = None

        for images, _, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            image_embeddings = model.image_encoder(images)
            image_embeddings = model.image_norm(image_embeddings)
            similarities = (text_embedding @ image_embeddings.T).squeeze(0)
            max_idx = similarities.argmax().item()
            if similarities[max_idx] > best_similarity:
                best_similarity = similarities[max_idx]
                best_match_idx = max_idx
                best_match_image = images[max_idx].cpu().squeeze(0)

    # Display result
    plt.imshow(best_match_image, cmap="gray")
    plt.title(f"Best Match for '{text_label}'")
    plt.axis("off")
    plt.show()

def infer_image_to_text(image_tensor):
    """
    Given an image, predict the closest matching text label.
    """
    image_tensor = image_tensor.to(device).unsqueeze(0)

    with torch.no_grad():
        image_embedding = model.image_encoder(image_tensor)
        image_embedding = model.image_norm(image_embedding)

        text_indices = torch.arange(len(digit_to_text), device=device)
        text_embeddings = model.text_embedding(text_indices)
        text_embeddings = model.text_norm(text_embeddings)

        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        predicted_idx = similarities.argmax().item()
        predicted_text = digit_to_text[predicted_idx]

    plt.imshow(image_tensor.cpu().squeeze(0), cmap="gray")
    plt.title(f"Predicted Label: {predicted_text}")
    plt.axis("off")
    plt.show()

model = ...
model.eval()
