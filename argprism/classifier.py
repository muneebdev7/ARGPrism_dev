"""
Neural Network Classifier for ARG prediction
"""

import torch


class ARGClassifier(torch.nn.Module):
    """
    Feed-forward neural network classifier for ARG prediction.
    
    Architecture:
    - Input: 4096-dimensional protein embeddings (from ProtAlbert)
    - Hidden layer 1: 512 neurons with ReLU
    - Hidden layer 2: 128 neurons with ReLU
    - Output: 2 classes (Non-ARG=0, ARG=1) with Softmax
    """
    
    def __init__(self, input_dim=4096):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 128)
        self.relu2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(128, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4096)
            
        Returns:
            Softmax probabilities of shape (batch_size, 2)
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(x)
        return self.softmax(x)
