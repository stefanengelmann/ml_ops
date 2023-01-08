from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifierNet = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3), # [batch_size, 64, 26, 26]
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3), # [batch_size, 32, 32, 24]
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3), # [batch_size, 16, 22, 22]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3), # [batch_size, 8, 20, 20]
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(8*20*20,256),
            #nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,10)
        )

    def forward(self, x):
        return self.classifierNet(x)

