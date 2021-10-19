import torch.nn as nn


class ClassifySoftmax(nn.Module):
    def __init__(self, num_inputs, num_class):
        super(ClassifySoftmax, self).__init__()
        self.output = nn.Linear(num_inputs, num_class)

    def forward(self, x):
        x = self.output(x)
        return x


def classify_softmax(num_inputs=100, num_class=2):
    return ClassifySoftmax(num_inputs=num_inputs, num_class=num_class)
