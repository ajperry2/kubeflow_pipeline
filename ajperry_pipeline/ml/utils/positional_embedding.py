import numpy as np
import torch


# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
  """
  Generate equally distant angle measurements
  """
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_embedding(position, d_model):

  """
  Generate positional embeddings, 
  typically used to embed the position of each token inputted into
  transformer architectures.
  """
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[None,...]
    
  return torch.tensor(pos_encoding)