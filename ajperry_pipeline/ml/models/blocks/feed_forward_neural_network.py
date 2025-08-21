import torch


class FeedForwardNeuralNetwork(torch.nn.Module):
    """
    The Neural Network block of a transformer architecture.
    
    It uses dropout and leaky relu

    Attributes:
        linears (list[Linear]): Self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """
    def __init__(self, num_layers: int, token_dim: int, dropout_p = 0.0, device: str = "cpu"):
        super().__init__()
        self.linears = [torch.nn.Linear(token_dim,token_dim).to(device) for i in range(num_layers)]
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.device = device
        
    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
            x = self.leaky_relu(x)
            if self.dropout_p > 0:
                x = self.dropout(x)
        return x