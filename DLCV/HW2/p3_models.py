import torch.nn as nn
import torch.nn.functional as F
# from utils import ReverseLayerF
import torch
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class FeatureExtractor(nn.Module):
    def __init__(self, in_chans=3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.reshape(-1, 128)
        return feature


class LabelPredictor(nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        self.l_clf = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.l_clf(x)
        return x


class DomainClassifier(nn.Module):
    '''
    A Binary classifier
    '''

    def __init__(self) -> None:
        super().__init__()
        self.d_clf = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

    def forward(self, x, lambda_):
        x = ReverseLayerF.apply(x, lambda_)
        x = self.d_clf(x)
        return x


if __name__ == '__main__':
    n = FeatureExtractor()
    n(torch.rand(64, 3, 32, 32))