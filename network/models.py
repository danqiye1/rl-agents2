import torch
import torch.nn as nn
from torchvision import models

class SqueezeNetQValCritic(nn.Module):
    """ 
    Function approximator of Q(s,a) for a Deep Q Learning task. This is implemented
    with a SqueezeNet. It is in the form of a Q-value critic.
    
    The neural network maps a state s, and an action a, to a Q value estimate.
    """

    def __init__(self, height, width, nch, n_actions):
        super(SqueezeNetQValCritic, self).__init__()

        # Main model is actually a squeezenet.
        # This converts image tensors of size (3, H, W) to (1000, 1) linear tensors
        self.squeezenet = models.squeezenet1_1()

        # Dimension check for linear layers with random tensor
        random_tensor = torch.rand([1, nch, height, width])
        output_tensor = self.squeezenet(random_tensor)
        assert output_tensor.size() == torch.Size([1, 1000])
        linear_size = output_tensor.size(1)

        # Linear layers for functional approximation
        self.linear = nn.Sequential(
            nn.Linear(linear_size, n_actions),
        )

    def forward(self, img):
        """ Q(s,a) function approximation """
        linear_tensor = self.squeezenet(img)
        return self.linear(linear_tensor)


