import torch


class Projection(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection_layer(x), dim=-1)
