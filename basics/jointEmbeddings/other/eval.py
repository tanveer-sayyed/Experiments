def get_embedding(model, image_path, text):
    image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()
    tokens = model.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    input_ids, attention_mask = tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()

    with torch.no_grad():
        img_emb, text_emb = model(image, input_ids, attention_mask)

    return img_emb, text_emb

# Compute similarity
img_emb, text_emb = get_embedding(model, "new_image.jpg", "A cat playing with a ball")
similarity = torch.nn.functional.cosine_similarity(img_emb, text_emb)
print(f"Similarity: {similarity.item():.4f}")
