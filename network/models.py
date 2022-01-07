import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace as bp

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
        
        # Use Kaiming initialization
        self.squeezenet.apply(self.init_weights)
        self.linear.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        """ Kaiming uniform initialization for linear and conv2d layers"""
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            # Only linear and conv2d layers have weights
            nn.init.kaiming_uniform(layer.weight, a=2)
            layer.bias.data.fill_(0.01)

    def forward(self, img):
        """ Q(s,a) function approximation """
        linear_tensor = self.squeezenet(img)
        return self.linear(linear_tensor)


