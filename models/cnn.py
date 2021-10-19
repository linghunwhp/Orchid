import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_class=100, dropout=False, batch_norm=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)    # the first conv
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)            # the second conv
        self.tanh = nn.Tanh()                       # the activate function
        self.pool = nn.MaxPool2d(2, 2)              # the pooling function
        self.fc1 = nn.Linear(64*16*16, 128)        # the first full connected layer
        self.fc2 = nn.Linear(128, num_class)        # the second full connected layer

    def forward(self, x):
        x = self.conv1(x)                       # call the first conv
        x = self.tanh(x)                        # call the activate function
        x = self.conv2(x)                       # call the second conv
        x = self.tanh(x)                        # call the activate function
        x = self.pool(x)                        # call the pooling function
        x = x.view(x.size(0), -1)
        x = self.fc1(x)                         # the first full connected layer
        output = self.fc2(x)                    # the second full connected layer
        return output


def cnn(num_class=100):
    return CNN(num_class=num_class)