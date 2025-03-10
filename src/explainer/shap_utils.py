from itertools import combinations
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import numpy as np
import random
import colorsys
import os

import requests
import json
import base64
from typing import Optional, Callable, Any, List, Dict, Union, Tuple
from os import getenv


def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a base64-encoded string.
    
    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interact_with_ollama(
    prompt: Optional[str] = None, 
    messages: Optional[Any] = None,
    image_path: Optional[str] = None,
    model: str = 'openhermes2.5-mistral', 
    stream: bool = False, 
    output_handler: Callable[[str], None] = default_output_handler,
    api_url: Optional[str] = None
) -> str:
    """
    Sends a request to the Ollama API for chat or image generation tasks.
    
    Args:
    prompt (Optional[str]): Text prompt for generating content.
    messages (Optional[Any]): Structured chat messages for dialogues.
    image_path (Optional[str]): Path to an image for image-related operations.
    model (str): Model identifier for the API.
    stream (bool): If True, keeps the connection open for streaming responses.
    output_handler (Callable[[str], None]): Function to handle output messages.
    api_url (Optional[str]): API endpoint URL; if not provided, fetched from environment.

    Returns:
    str: Full response from the API.

    Raises:
    ValueError: If necessary parameters are not correctly provided or API URL is missing.
    """
    if not api_url:
        api_url = getenv('API_URL')
        if not api_url:
            raise ValueError('API_URL is not set. Provide it via the api_url variable or as an environment variable.')

    # Define endpoint based on whether it's a chat or a single generate request
    api_endpoint = f"{api_url}/api/{'chat' if messages else 'generate'}"
    data = {'model': model, 'stream': stream}

    # Populate the request data
    if messages:
        data['messages'] = messages
    elif prompt:
        data['prompt'] = prompt
        if image_path:
            data['images'] = [encode_image_to_base64(image_path)]
    else:
        raise ValueError("Either messages for chat or a prompt for generate must be provided.")

    # Send request
    response = requests.post(api_endpoint, json=data, stream=stream)
    if response.status_code != 200:
        output_handler(f"Failed to retrieve data: {response.status_code}")
        return ""

    # Handle streaming or non-streaming responses
    full_response = ""
    if stream:
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
                full_response += response_part
                output_handler(response_part)
                if json_response.get('done', False):
                    break
    else:
        json_response = response.json()
        response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
        full_response += response_part
        output_handler(response_part)

    return full_response, response

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

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

class HuggingFaceEmbeddings(TextVectorizer):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace sentence embeddings vectorizer - much simpler implementation
        
        Args:
            model_name: Name of the sentence-transformer model from HuggingFace
            device: Device to run model on ('cpu' or 'cuda', etc.)
        """
        self.model_name = model_name
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


class Splitter:
    def split(self, prompt):
        raise NotImplementedError

    def join(self, tokens):
        raise NotImplementedError

class StringSplitter(Splitter):
    def __init__(self, split_pattern = r'\b\w+\b'):
        self.split_pattern = split_pattern
    
    def split(self, prompt):
        return [word for word in re.findall(self.split_pattern, prompt.strip()) if word]
    
    def join(self, tokens):
        return ' '.join(tokens)

class TokenizerSplitter(Splitter):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split(self, prompt):
        return self.tokenizer.tokenize(prompt)

    def join(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)
    
