import torch
import torch.nn as nn
from torchvision import models

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Build layers from configuration list.
    'M' = MaxPool
    Others = Conv2d + ReLU
    """
    layers = []
    d_rate = 2 if dilation else 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # Frontend configuration (VGG16 style)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Backend configuration (dilated conv)
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        # Build layers
        self.frontend = make_layers(self.frontend_feat)
        self.backend  = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

        # Load VGG16 pretrained weights into frontend if requested
        if not load_weights:
            self._load_vgg16_frontend_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_vgg16_frontend_weights(self):
        """
        Copy weights from pretrained VGG16 into frontend layers.
        """
        vgg16 = models.vgg16(pretrained=True)
        # Use zip to copy layer weights safely
        for (target_name, target_param), (source_name, source_param) in zip(
                self.frontend.state_dict().items(),
                vgg16.features.state_dict().items()):
            target_param.data.copy_(source_param.data)