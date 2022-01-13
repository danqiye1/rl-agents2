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

        # Main model is actually a squeezenet, but modified to fit number of input channels
        # This converts image tensors of size (nch, H, W) to (1000, 1) linear tensors
        self.squeezenet = models.squeezenet1_1()
        self.squeezenet.features[0] = nn.Conv2d(nch, 64, kernel_size=(3, 3), stride=(2, 2))

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
            nn.init.kaiming_uniform_(layer.weight, a=2)
            layer.bias.data.fill_(0.01)

    def forward(self, img):
        """ Q(s,a) function approximation """
        linear_tensor = self.squeezenet(img)
        return self.linear(linear_tensor)

class MinhDQN(nn.Module):
    """
    Vanilla Deep Q Network as described in Minh. et. al (2013).
    This network requires all images to be preprocessed to 84 x 84, stacked to k = 4 or k = 3 frames,
    as described in the paper. Therefore the input image is expected to be 84 x 84 x k.

    Implemented for benchmarking and sanity check.
    """

    def __init__(self, nch, n_actions):
        super(MinhDQN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(nch, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )

        # Dimension calculation
        random_tensor = torch.rand([1, nch, 84, 84])
        output_tensor = self.cnn(random_tensor)
        linear_size = torch.flatten(output_tensor).size(0)

        self.linear = nn.Sequential(
            nn.Linear(linear_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

        # Use Kaiming initialization
        self.cnn.apply(self.init_weights)
        self.linear.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        """ Kaiming uniform initialization for linear and conv2d layers"""
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            # Only linear and conv2d layers have weights
            nn.init.kaiming_uniform_(layer.weight, a=2)
            layer.bias.data.fill_(0.01)

    def forward(self, img):
        """ Q(s,a) function approximation """
        linear_tensor = torch.flatten(self.cnn(img), start_dim=1)
        return self.linear(linear_tensor)


