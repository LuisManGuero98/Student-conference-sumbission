import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim
from torchrl.modules import ValueOperator
from torch import multiprocessing


'''
**************************************************************************

Contains CNNs classes to be used by PPO method.

**************************************************************************
'''


class PPO_UNet(nn.Module):

    '''
    From diffusion model with U-Net architecture as mentioned in paper
    U-Net architecture, returns the 7 output from the CNN (joints positions)

        Encoders(image) -> Bottleneck(last encoder) -> Decoders(bottleneck) -> Final layer

        Skip features (SF)

        image-> Encode                SF->                  Decode  -> Final layer (7 joints)
                    Encode            SF->              Decode
                        Encode        SF->         Decode
                            Encode    SF->      Decode
                                    Bottleneck
    '''

    def __init__(self, input_dim = 3, output_dimension = 7):

        super().__init__() # Instantiate nn.module

        self.checkpoint = 'UNet_checkpoint'

        self.input_dim = input_dim
        self.output_dim = output_dimension

        self.utils = Utils()
        # self.save = self.utils.save_checkpoint(self.checkpoint)
        # self.load = self.utils.load_checkpoint(self.checkpoint)

        self.features = [64, 128, 256, 512]
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork # For Linux
            else torch.device("cpu")
        )

        self.to(self.device)

        # Initialize contracting layers (3 -> 64 -> 128 -> 256 -> 512 -> 1024)
        inp = input_dim
        for feature in self.features:
            self.encoders.append(self.doubleConv(inp, feature))
            inp = feature

        # Initialize expansive layers (opposite as extracting), adding skip connections
        # that link feature maps from encoder and decoder at corresponging layers
        for feature in reversed(self.features):
            self.decoders.append(self.expanding_block(feature * 2, feature))
            self.decoders.append(self.doubleConv(feature * 2, feature)) # 2 convs up

        # Initialize bottleneck
        self.bottleneck = self.doubleConv(self.features[-1], self.features[-1] * 2) # 1024

        # Final layer (outputs 7 joints)
        self.final_layer = nn.Conv2d(self.features[0], self.output_dim, kernel_size=1)

    def expanding_block(self, input_size, output_size):

        # Expanding path
        return nn.Sequential(
            nn.ConvTranspose2d(input_size, output_size, kernel_size = 2, stride = 2, padding = 1, bias = False),
            nn.ReLU(inplace = True)
        )

    def doubleConv(self, input, output):

        # Performs double convolution (as in the paper)
        return nn.Sequential(

            nn.Conv2d(input, output, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(output),     # Batch Normalization over a 4D input.
            nn.ReLU(inplace=True),

            nn.Conv2d(output, output, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        skip_connections = []

        # Asegurar que el tensor tiene 4 dimensiones
        if x.dim() == 3:  # Si tiene solo [height, width, channels]
            x = x.unsqueeze(0)  # Agregar batch dimension -> [1, height, width, channels]

        # Asegurar que los canales están en la segunda dimensión
        if x.shape[-1] == 3:  # Si los canales están en la última dimensión
            x = x.permute(0, 3, 1, 2)  # Convertir a [batch, channels, height, width]

        # Contracting path
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck features
        y = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # List in reverse

        # Expanding path
        for i in range(0, len(self.decoders), 2):

            y = self.decoders[i](y)        # 1 Decoding
            skip = skip_connections[i // 2]         # 2 Obtain skip connection
            if y.shape != skip.shape:               # in case i // 2 not precise
                y = tf.resize(y, size = skip.shape[2:])
            conc = torch.cat((skip, y), dim = 1)    # 3 Concatenate skip with decoded
            y = self.decoders[i + 1](conc)          # 4 Decoding with concatenate

        y = self.final_layer(y)
        scale = torch.std(y, dim=(1,2,3), keepdim=True).flatten()
        loc = torch.mean(y, dim=(2,3)).flatten()

        return {"loc": loc, "scale": scale}

class value_network(nn.Module):

    def __init__(self, input_channels=3, output_dim=1, device="cuda"):
        super(value_network, self).__init__()

        self.device = device
        self.to(self.device)

        # Convolutional Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),  # 256x256 → 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64 → 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32 → 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16 → 8x8
            nn.ReLU(),
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flattens output from CNN
            nn.Linear(256 * 8 * 8, 512),  # Adjust based on CNN output size
            nn.ReLU(),
            nn.Linear(512, output_dim)  # Outputs a single value (state value)
        )

    def forward(self, x):

        self.to(self.device)

        if len(x.shape) == 5:  
            # Ensure it's a 4D tensor: (batch, channels, height, width), remove
            # extra dimension if needed.
            x = x.squeeze(1)  

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

    def value_module(self):
        """ Returns the model itself for external use """
        return self

class Utils(nn.Module):

    '''
        Save and load of NNs
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_checkpoint(self, checkpoint):
        print('Saving checkpoint')
        torch.save(self.state_dict(), checkpoint)

    def load_checkpoint(self, checkpoint):
        print('Loading checkpoint')
        self.load_state_dict(torch.load(checkpoint))


def test():

    '''
        Tests to confirm UNet is working correctly
        (programming purposes)
    '''

    x = torch.randn((3,1,160,160))
    model = PPO_UNet(1,1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

'''
    Auxiliary inputs for UNet
'''
# Links to mannequin distances as auxiliary input branch
# (Currently not using)
# distances = 7 # for each joint
# self.aux_input_branch = nn.Sequential(
#     nn.Linear(distances, 32),
#     nn.ReLU(),
#     nn.Linear(32,64),
#     nn.ReLU()
# )

# aux = self.aux_input_branch(distances)
# aux = aux.unsqueeze(-1).unsqueeze(-1)  # Reshape for concatenation
# Combined features
# combined = torch.cat([bottleneck, aux.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))], dim=1)
# y = self.decoders[0](combined)
# for i in range(1, len(self.decoders), 2):