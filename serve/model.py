import torch.nn as nn
import torch.nn.functional as F


class DataDiscriminator(nn.Module):
    
    def __init__(self, hidden_dim, dropout =0.3):
        
        super(DataDiscriminator, self).__init__()
        
        self.fc1 = nn.Linear(30, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, 32)
        
        self.fc5 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
               
        out = self.dropout(F.relu(self.fc1(x)))
        
        out = self.dropout(F.relu(self.fc2(out)))
        
        out = self.dropout(F.relu(self.fc3(out)))
        
        out = self.dropout(F.relu(self.fc4(out)))
        
        out = self.fc5(out)
        
        return out
    

class DataGenerator(nn.Module):
    
    def __init__(self, hidden_dim, z_size, dropout =0.3):
        super(DataGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(z_size, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim* 3)
        
        self.fc4 = nn.Linear(hidden_dim* 3, hidden_dim * 2)
        
        self.fc5 = nn.Linear(hidden_dim * 2, 30)
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        
        out = self.dropout(F.leaky_relu(self.fc1(x)))
        
        out = self.dropout(F.leaky_relu(self.fc2(out)))
        
        out = self.dropout(F.leaky_relu(self.fc3(out)))
        
        out = self.dropout(F.leaky_relu(self.fc4(out)))
        
        out = self.fc5(out)
        
        return out