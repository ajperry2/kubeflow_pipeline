import torch


class FeedForwardNeuralNetwork(torch.nn.Module):
    """
    The Neural Network block of a transformer architecture.

    It uses dropout and leaky relu

    Attributes:
        linears (list[Linear]): Self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
