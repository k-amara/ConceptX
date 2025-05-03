from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

class TextVectorizer:
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text strings into vector representations.
        
        Args:
            texts: List of text strings to vectorize
            
        Returns:
            numpy array of vectors
        """
        raise NotImplementedError
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between vectors
        
        Args:
            base_vector: Vector to compare against
            comparison_vectors: List of vectors to compare with base_vector
            
        Returns:
            numpy array of similarity scores
        """
        raise NotImplementedError

#"sentence-transformers/all-mpnet-base-v2"
class HuggingFaceEmbeddings(TextVectorizer):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace sentence embeddings vectorizer - much simpler implementation
        
        Args:
            model_name: Name of the sentence-transformer model from HuggingFace
            device: Device to run model on ('cpu' or 'cuda', etc.)
        """
        self.model_name = model_name
        print("Embedding Model: ", model_name)
        self.device = device
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load model - SentenceTransformer handles all the complexity
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers - much simpler"""
        if not self.model:
            self._initialize_model()
            
        # SentenceTransformer handles batching, padding, etc. automatically
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Sentence-transformers models already return normalized vectors
        return np.dot(comparison_vectors, base_vector)

class OpenAIEmbeddings(TextVectorizer):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings vectorizer
        
        Args:
            api_key: OpenAI API key
            model: Embeddings model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        if not self.client:
            self._initialize_client()
            
        # Process texts in batches to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                # Extract embeddings from response
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise Exception(f"Error getting embeddings from OpenAI: {str(e)}")
                
        return np.array(all_embeddings)
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Normalize vectors
        base_norm = np.linalg.norm(base_vector)
        comparison_norms = np.linalg.norm(comparison_vectors, axis=1)
        
        # Avoid division by zero
        if base_norm == 0 or np.any(comparison_norms == 0):
            return np.zeros(len(comparison_vectors))
            
        normalized_base = base_vector / base_norm
        normalized_comparisons = comparison_vectors / comparison_norms[:, np.newaxis]
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_comparisons, normalized_base)
        return similarities

class TfidfTextVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = None
        
    def vectorize(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer().fit(texts)
        return self.vectorizer.transform(texts).toarray()
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            base_vector.reshape(1, -1), comparison_vectors
        ).flatten()

# OpenAI Embeddings implementation
class OpenAIEmbeddings(TextVectorizer):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings vectorizer
        
        Args:
            api_key: OpenAI API key
            model: Embeddings model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        if not self.client:
            self._initialize_client()
            
        # Process texts in batches to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                # Extract embeddings from response
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise Exception(f"Error getting embeddings from OpenAI: {str(e)}")
                
        return np.array(all_embeddings)
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Normalize vectors
        base_norm = np.linalg.norm(base_vector)
        comparison_norms = np.linalg.norm(comparison_vectors, axis=1)
        
        # Avoid division by zero
        if base_norm == 0 or np.any(comparison_norms == 0):
            return np.zeros(len(comparison_vectors))
            
        normalized_base = base_vector / base_norm
        normalized_comparisons = comparison_vectors / comparison_norms[:, np.newaxis]
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_comparisons, normalized_base)
        return similarities