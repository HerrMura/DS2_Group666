from torch import nn

class SROP(nn.Module):
    def __init__(self, num_channels=3,scale=2):
        super(SROP, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=scale),
            nn.Conv2d(16, 2, kernel_size=5, padding=5 // 2)
            #nn.ReLU() # the last relu will reduce the distortion but also the anti-aliasing effect on edges
        )

    def forward(self, x):
        x = self.model(x)
        return x 
