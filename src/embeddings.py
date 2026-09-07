from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "The tenant must provide two months written notice.",
    "How long before leaving should I notify the landlord?",
    "The apartment has two bedrooms and a balcony."
]


embeddings = embedding_model.embed_documents(texts)


similarity_1 = np.dot(embeddings[0], embeddings[1])
similarity_2 = np.dot(embeddings[0], embeddings[2])


print("Similarity between sentence 1 and 2:", similarity_1)
print("Similarity between sentence 1 and 3:", similarity_2)