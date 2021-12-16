from torch import nn


class Classifier(nn.Module):
    def __init__(self, num_channels, num_classes, train_backbone) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

        if not train_backbone:
            for name, parameter in self.linear.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.linear(out.squeeze())  # No batch_size=1
        out = self.sigmoid(out)
        return out


def build_classifier(config):
    train_backbone = config.lr_backbone > 0
    num_channels = 512 if config.backbone in ('resnet18', 'resnet34') else 2048
    classifier = Classifier(num_channels, config.num_classes, train_backbone)
    return classifier
