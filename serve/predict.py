import argparse
import json
import os
import pickle
import sys
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pickle as pkl

from model import DataDiscriminator, DataGenerator

from utils import string_to_num

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #D = DataDiscriminator(64)
    G = DataGenerator(z_size=100, hidden_dim=64)
    
    model_info = {}
    model_info_path = os.path.join(model_dir, 'generator_model.pt')
    
    print("Inside Model")
    print(model_info_path)
    print(os.listdir(model_dir))
    
    with open(model_info_path, 'rb') as f:
        G.load_state_dict(torch.load(f))

    G.to(device).eval()

    print("Done loading model.")
    return G


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    
    print(serialized_input_data)
    
    print(content_type)
    
    if content_type == 'text/plain':
        
        data = serialized_input_data.decode('utf-8')
        
        print(data)
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    
    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    
    return str(prediction_output)



def predict_fn(input_data, model):
    print('Generating Synthetic data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    input_noise = string_to_num(input_data)
      
    input_noise = input_noise.to(device)
    
    print("INput noise")
    print(input_noise)
    
    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    
    # data = input_noise
    # data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    
    result = model(input_noise).detach()

    return result


    
    
