import openai
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import json

clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# Sample knowledge base (can be extended)
knowledge_base = [
    "A cat is a small domesticated carnivorous mammal.",
    "A dog is a domesticated carnivore of the family Canidae.",
    "An apple is a sweet, edible fruit produced by an apple tree.",
    "A car is a wheeled motor vehicle used for transportation."
]

# Encode knowledge base into FAISS
kb_embeddings = sbert.encode(knowledge_base, convert_to_numpy=True)
index.add(kb_embeddings)

def retrieve_knowledge(description):
    query_vector = sbert.encode([description], convert_to_numpy=True)
    _, indices = index.search(query_vector, k=2)
    return [knowledge_base[i] for i in indices[0]]

def classify_image(image_path, label_chunks):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(clip_model.visual.input_resolution)
    
    similarities = []
    for chunk in label_chunks:
        text_inputs = clip.tokenize(chunk).to(clip_model.visual.input_resolution)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text_inputs)
        similarities.extend((image_features @ text_features.T).squeeze().tolist())
    
    best_match_idx = similarities.index(max(similarities))
    labels = sum(label_chunks, [])  # Flatten list of chunks
    predicted_label = labels[best_match_idx]
    
    return predicted_label

def chunk_labels(labels, chunk_size=75):
    return [labels[i:i+chunk_size] for i in range(0, len(labels), chunk_size)]

def react_agent(image_path, labels):
    """Implements ReAct framework for reasoning and acting with a loop."""
    label_chunks = chunk_labels(labels)
    confidence_threshold = 0.95  # Define a confidence threshold
    
    while True:
        # Step 1: Observe
        predicted_label = classify_image(image_path, label_chunks)
        print(f"Current Prediction: {predicted_label}")
        
        # Step 2: Retrieve relevant knowledge
        retrieved_facts = retrieve_knowledge(predicted_label)
        print(f"Retrieved Knowledge: {retrieved_facts}")
        
        # Step 3: Reasoning with LLM
        prompt = f"""
        You are an AI assistant helping with image classification.
        The model predicts the image as '{predicted_label}'.
        Based on the following facts, verify if the prediction is correct:
        {json.dumps(retrieved_facts)}
        Provide a final label if needed and a confidence score (0 to 1)."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an AI reasoning assistant."},
                      {"role": "user", "content": prompt}]
        )
        result = response["choices"][0]["message"]["content"].strip()
        
        # Extract refined label and confidence score
        try:
            refined_label, confidence = result.rsplit(" ", 1)
            confidence = float(confidence)
        except:
            refined_label, confidence = result, 0.5  # Default confidence if parsing fails
        
        print(f"Refined Label: {refined_label} (Confidence: {confidence})")
        
        # Stop if confidence is high enough
        if confidence >= confidence_threshold:
            break
    
    return refined_label

# Example usage
image_path = "example.jpg"  # Replace with actual image path
labels = ["cat", "dog", "apple", "car", "airplane", "bicycle", "horse", "zebra", "lion", "truck", "tree", "building"]  # Extended labels
final_label = react_agent(image_path, labels)
print(f"Labeled Image as: {final_label}")

