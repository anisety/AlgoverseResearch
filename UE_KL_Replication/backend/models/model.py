import torch
import torch.nn as nn

class SimpleUEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleUEModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits
