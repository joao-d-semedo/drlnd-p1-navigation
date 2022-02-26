import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.layers = []
        for i in range(len(sizes)-2):
            self.layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
        self.layers.append( nn.Linear(sizes[-2], sizes[-1]) )
        if output_activation is not None:
            self.layers.append(output_activation())
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNet(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=32, kernel_size=8, stride=4) #  3x84x84 -> 32x20x20
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # 32x20x20 -> 64x 9x 9
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 64x 9x 9 -> 64x 7x 7
        self.linear1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.linear2 = nn.Linear(in_features=   512, out_features=output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = x.permute((0, 3, 1, 2)) # Channels first
        batch_size = x.shape[0]
        x = self.activation( self.conv1(x) )
        x = self.activation( self.conv2(x) )
        x = self.activation( self.conv3(x) )
        x = x.reshape( (batch_size,-1) )
        x = self.activation( self.linear1(x) )
        x = self.activation( self.linear2(x) )
        return x


    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        if isinstance(state_size, tuple):
            self.net = ConvNet(action_size)
        else:
            hidden_sizes = [64, 64]
            self.net = MLP( sizes=[state_size] + hidden_sizes + [action_size] )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.net(state)
