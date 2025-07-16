import torch.nn as nn

class UNetPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPP, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x), x[0, 0].detach().numpy()
