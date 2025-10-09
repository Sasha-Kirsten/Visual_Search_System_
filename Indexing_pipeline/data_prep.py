import numpy as np
from ML_Training.MODEL.generating_embeddings
from ML_Training.MODEL.model

your_embedding =  generating_embeddings()
# Example: You have a list of embeddings from a model like Sentence-BERT
# your_embeddings is a list of lists, or a 2D array from a model
data_vectors = np.array(your_embeddings).astype('float32')

# Verify the shape and type
print(data_vectors.shape) # Output: (num_vectors, dimensionality)
print(data_vectors.dtype) # Output: float32