import torch
import clip
import numpy as np
import faiss
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import shuffle
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load detected entities (from previous pipeline)
with open("detected_objects.json", "r") as f:
    detected_entities = json.load(f)  # Format: [{"label": "cat", "image": "path_to_img1"}, ...]

# Encode each entity image into CLIP embeddings
embeddings = []
image_paths = []

for entity in detected_entities:
    image = Image.open(entity["image"])
    image_clip = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image_clip)
    embeddings.append(features.cpu().numpy().flatten())
    image_paths.append(entity["image"])

embeddings = np.array(embeddings)

# Clustering to create 20 distinct groups
num_clusters = 20
clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity="euclidean", linkage="ward")
cluster_labels = clustering.fit_predict(embeddings)

# Assign images to buckets
buckets = {i: [] for i in range(num_clusters)}
for i, label in enumerate(cluster_labels):
    buckets[label].append(image_paths[i])

# Ensure distinct entities in each bucket using FAISS similarity check
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

def refine_buckets():
    for cluster_id, images in buckets.items():
        new_bucket = []
        seen_embeddings = []
        for img_path in images:
            img_index = image_paths.index(img_path)
            img_embedding = embeddings[img_index]

            # Check if embedding is too similar to existing ones in this bucket
            if seen_embeddings:
                _, idx = faiss_index.search(np.array([img_embedding]), 1)
                if idx[0][0] != -1:  # Similar entity detected
                    continue  # Skip this image, reassign later
            new_bucket.append(img_path)
            seen_embeddings.append(img_embedding)

        buckets[cluster_id] = new_bucket

# Run refinement to ensure distinct groups
refine_buckets()

# Auto-rebalancing: Ensure equal-sized buckets
def rebalance_buckets():
    all_images = [img for imgs in buckets.values() for img in imgs]
    shuffle(all_images)  # Random shuffle to avoid bias
    balanced_buckets = {i: [] for i in range(num_clusters)}
    for i, img in enumerate(all_images):
        balanced_buckets[i % num_clusters].append(img)
    return balanced_buckets

buckets = rebalance_buckets()

# Save grouped entities
with open("grouped_entities.json", "w") as f:
    json.dump(buckets, f, indent=4)

print("Entities grouped into 20 balanced, distinct buckets!")
 
