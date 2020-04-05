import numpy as np
import pickle

import os
import glob
import torch


def string_to_num(z_size_data):
    
    z_size_data = str(z_size_data)
    
    z_size_data = [float(i) for i in z_size_data.split(',')]
    
    z_size_data = torch.tensor(z_size_data)
    
    print("Inside Util")
    
    return z_size_data.detach()