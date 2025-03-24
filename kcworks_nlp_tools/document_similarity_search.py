from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from typing import List, Tuple
import pickle

# Ideas from this thread:
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
#
# Remember to:
# Use batch processing to manage memory
# Consider using a more sophisticated FAISS index type (like IVF) for larger datasets
# Monitor memory usage and adjust batch sizes accordingly
# Use GPU acceleration if available
# Consider implementing parallel processing for the indexing phase


class DocumentSearchEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize FAISS index
        # 384 is the dimension of embeddings from all-MiniLM-L6-v2
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)

        # Store document mapping
        self.documents = []

    def index_documents_from_file(self, file_path: str, batch_size: int = 32):
        """Index documents from a large file without loading all into memory"""

        def document_generator(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    yield line.strip()

        # Count total documents for progress tracking
        total_docs = sum(1 for _ in open(file_path, "r"))

        batch = []
        for i, doc in enumerate(document_generator(file_path)):
            batch.append(doc)

            if len(batch) >= batch_size:
                batch_embeddings = np.vstack([self.get_embedding(d) for d in batch])
                self.index.add(batch_embeddings)
                self.documents.extend(batch)
                batch = []

                if i % 1000 == 0:
                    print(
                        f"Indexed {i}/{total_docs} documents "
                        f"({((i/total_docs)*100):.2f}%)"
                    )

        # Index remaining documents
        if batch:
            batch_embeddings = np.vstack([self.get_embedding(d) for d in batch])
            self.index.add(batch_embeddings)
            self.documents.extend(batch)

    def optimize_index(self):
        """Optional: Convert to IVF index for faster searching with large datasets"""
        nlist = min(4096, len(self.documents))  # number of clusters
        quantizer = faiss.IndexFlatIP(self.dimension)
        new_index = faiss.IndexIVFFlat(
            quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )

        # Train the index
        train_vectors = np.vstack(
            [
                self.get_embedding(doc)
                for doc in self.documents[: min(100000, len(self.documents))]
            ]
        )
        new_index.train(train_vectors)

        # Add all vectors
        new_index.add(np.vstack([self.get_embedding(doc) for doc in self.documents]))

        self.index = new_index

    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take attention mask into account for correct averaging
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embedding(self, text: str) -> np.ndarray:
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        embeddings = self.mean_pooling(outputs, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def index_documents(self, documents: List[str], batch_size: int = 32):
        """Index documents in batches to handle large collections"""
        self.documents = documents

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_embeddings = np.vstack([self.get_embedding(doc) for doc in batch])
            self.index.add(batch_embeddings)

            if i % 1000 == 0:
                print(f"Indexed {i} documents...")

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)

        # Return results with documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Ensure valid index
                results.append((idx, float(score), self.documents[idx]))

        return results

    def save_index(self, index_path: str, documents_path: str):
        """Save the FAISS index and documents to disk"""
        faiss.write_index(self.index, index_path)
        with open(documents_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, index_path: str, documents_path: str):
        """Load the FAISS index and documents from disk"""
        self.index = faiss.read_index(index_path)
        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)


# Example usage with large dataset
def process_large_dataset():
    search_engine = DocumentSearchEngine()

    # Index documents from file
    search_engine.index_documents_from_file("large_document_collection.txt")

    # Optionally optimize the index for faster searching
    search_engine.optimize_index()

    # Save the optimized index
    search_engine.save_index("optimized_index.faiss", "documents.pkl")


# Example usage:
def main():
    # Initialize the search engine
    search_engine = DocumentSearchEngine()

    # Example documents (replace with your 2M documents)
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing deals with text analysis",
        # ... more documents ...
    ]

    # Index the documents
    print("Indexing documents...")
    search_engine.index_documents(documents)

    # Save the index for later use
    print("Saving index...")
    search_engine.save_index("document_index.faiss", "documents.pkl")

    # Example search
    query = "What is AI and machine learning?"
    results = search_engine.search(query, k=3)

    print(f"\nSearch results for: '{query}'")
    for idx, score, doc in results:
        print(f"Score: {score:.4f} - {doc}")


if __name__ == "__main__":
    main()
