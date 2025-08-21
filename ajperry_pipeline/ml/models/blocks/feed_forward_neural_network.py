import torch


class FeedForwardNeuralNetwork(torch.nn.Module):
    def __init__(self, num_layers: int, token_dim: int, dropout_p = 0.0):
        super().__init__()
        self.linears = [torch.nn.Linear(token_dim,token_dim) for i in range(num_layers)]
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
            x = self.leaky_relu(x)
            if self.dropout_p > 0:
                x = self.dropout(x)
        return x