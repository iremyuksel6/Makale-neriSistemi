import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

def get_scibert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def compute_cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def get_average_embedding(words):
    embeddings = [get_scibert_embeddings(word) for word in words]
    average_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return average_embedding

makale_vektorleri = np.load('makale_vektorleri.npy', allow_pickle=True).item()

words = ["problem", "solution", "research"]
word_vector = get_average_embedding(words)

similarities = {}
for key, vec in makale_vektorleri.items():
    vec_tensor = torch.tensor(vec)
    similarity = compute_cosine_similarity(word_vector, vec_tensor)
    similarities[key] = similarity

sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
top_5_similarities = sorted_similarities[:5]

for key, similarity in top_5_similarities:
    print(f"Makale: {key}, Benzerlik: {similarity}")
