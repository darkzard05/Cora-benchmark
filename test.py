import torch
import torch_geometric
from torch_geometric.nn import Linear

class model(torch.nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer = Linear(input, output)
        
    def reset_parameter(self):
        self.layer.reset_parameter()
        
    def forward(self, data):
        x = data
        y = self.layer(x)
        return torch.softmax(y)