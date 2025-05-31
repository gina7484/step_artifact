from torch import nn
from base import Stream


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Stream) -> Stream:
        # Implement the forward pass logic here
        pass
