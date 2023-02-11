import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels as ptm


class Densenet121(nn.Module):
    def __init__(self, pretrained):
        super(Densenet121, self).__init__()
        self.model = ptm.__dict__["densenet121"](
            num_classes=1000, pretrained="imagenet" if pretrained else None
        )

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x


class Network(nn.Module):
    def __init__(self, num_classes, args, pretrained=True):
        super().__init__()
        self.args = args
        self.encoder = Densenet121(pretrained=pretrained)

        self.feature_dim = self.encoder.model.last_linear.in_features

        self.fc = nn.Linear(self.feature_dim, args.dim_rep)
        self.out = nn.Linear(args.dim_rep, num_classes)

    def forward(self, x):
        fea = self.encoder(x)
        fea = F.normalize(self.fc(fea), dim=-1)
        out = self.out(fea)

        return out, fea
