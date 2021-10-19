import torch.nn as nn


class ClassifyNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_class):
        super(ClassifyNN, self).__init__()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(num_hiddens, num_class)

    def forward(self, x):
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        return x


def classify_nn(num_inputs=100, num_hiddens=50, num_class=2):
    return ClassifyNN(num_inputs=num_inputs, num_hiddens=num_hiddens, num_class=num_class)
