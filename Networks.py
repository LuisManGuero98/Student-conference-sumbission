import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.modules import ValueOperator


'''
**************************************************************************

Contains CNNs classes to be used by PPO method.

**************************************************************************
'''


class PPO_UNet(nn.Module):

    # From diffusion model with U-Net architecture as mentioned in paper
    # U-Net architecture, returns the 7 output from the CNN (joints positions)

    def __init__(self, input_dim, output_dimension, alpha):

        super().__init__() # Instantiate nn.module

        # Checkpoint
        self.checkpoint = 'UNet_checkpoint'

        # Auxiliary features: Distances end_effector2goal, links2person
        # auxiliary input branch
        self.aux_input_branch = nn.Sequential(
            nn.Linear(input_dim, 32), 
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU()
        )

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)

        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork # For Linux
            else torch.device("cpu")
        )

        self.to(self.device)

        # Initialize contracting layers (3 -> 64 -> 128 -> 256 -> 512 -> 1024)
        self.encoder_1 = self.contracting_block(3, 64)
        self.encoder_2 = self.contracting_block(64, 128)
        self.encoder_3 = self.contracting_block(128, 256)
        self.encoder_4 = self.contracting_block(256, 512)

        # Initialize bottleneck
        self.bottleneck = self.bottleneck_block(512,1024)

        # Initialize expansive layers (opposite as extracting)
        # Adding components are skip connections that link feature maps from encoder 
        # and decoder at corresponging layers
        self.decoder_1 = self.expanding_block(1024 + 64, 512) # include aux features
        self.decoder_2 = self.expanding_block(512 + 256, 256)
        self.decoder_3 = self.expanding_block(256 + 128, 128)
        self.decoder_4 = self.expanding_block(128 + 64, 128)

        # Final layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Reduce to (batch, 64, 1)
        self.final_layer = nn.Linear(64 * 8 * 8, output_dimension)  # 64 -> 7
        
    def contracting_block(self, input_size, output_size):

        # Contracting path: two 3x3 - ReLU - 2x2 max pooling
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size = 3, padding = 1), nn.ReLU(inplace = True),
            nn.Conv2d(output_size, output_size, kernel_size = 3, padding = 1), nn.ReLU(inplace = True)
        )
    
    def bottleneck_block(self, input_size, output_size):

        # Bottleneck between contracting and expanding
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size = 3, padding = 1), nn.ReLU(inplace = True),
            nn.Conv2d(output_size, output_size, kernel_size = 3, padding = 1), nn.ReLU(inplace = True)
        )
    
    def expanding_block(self, input_size, output_size):

        # Expanding path
        return nn.Sequential(
            nn.ConvTranspose2d(input_size, output_size, kernel_size = 2, padding = 2), nn.ReLU(inplace = True)
        )
    
    def forward(self, image, distances): # what distances?
        
        # contracting path 
        x_1 = self.encoder_1(image)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        # Bottleneck features
        bottleneck = self.bottleneck(x_4)

        # auxiliary inputs
        aux = self.aux_input_branch(distances)
        aux = aux.unsqueeze(-1).unsqueeze(-1)  # Reshape for concatenation

        # Combined features
        combined = torch.cat([bottleneck, aux.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))], dim=1)

        # Expanding path
        y = self.decoder_1(combined)
        y = torch.cat([y,x_4], dim = 1)
        y = self.decoder_2(y)
        y = torch.cat([y,x_3], dim = 1)
        y = self.decoder_3(y)
        y = torch.cat([y,x_2], dim = 1)
        y = self.decoder_4(y)
        y = torch.cat([y,x_1], dim = 1)

        # Flatten and predict trajectory
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1) # Flatten
        
        return self.final_layer(y)
    
    def save_checkpoint(self):
        torch.save(self.state_dict() , self.checkpoint)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))


class value_network(nn.Module):

    def __init__(self, num_cells, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.value_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

        self.value_module = ValueOperator(
            module = self.value_net,
            in_keys = ["observation"]
        )
