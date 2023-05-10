import typing as t
import torch
import torch.nn as nn


class Network(nn.Module):
    """State-Action Value Approximation (Q-Network)"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: t.Tuple[int, ...],
        seed: int = 5,
    ) -> None:
        """__init__

        Initialize a linear MLP with RELU activation functions.

        Parameters
        ----------
        state_size : int
            size of input state
        action_size : int
            size of output (number of available actions)
        hidden_sizes : t.Tuple[int, ...]
            hidden layer sizes
        seed : int, optional
            random seed for network, by default 5
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = (state_size,) + hidden_sizes + (action_size,)
        self.model = self.build_model(layer_sizes)

    def build_model(self, layer_sizes: t.Tuple[int, ...]) -> None:
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(
                nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
            )

            # Last layer no activation function
            if i != len(layer_sizes) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.model(state)
