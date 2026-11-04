"""my_torch: generic neural network module (creation, training, saving/loading)."""

from my_torch.errors import MyTorchError
from my_torch.layers import Dense

__all__ = ["Dense", "MyTorchError"]
