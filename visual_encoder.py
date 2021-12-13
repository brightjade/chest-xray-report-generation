import torch
import timm

if __name__ == '__main__':
    # Example usage
    num_classes = 10
    seresnet = timm.create_model('seresnet50', True, num_classes=num_classes)
    print(seresnet)
    x = torch.randn((2, 3, 512, 512))  # B, C, H, W
    out = seresnet(x)
    print(out.shape)  # not sum-to-one

    # multi-label classification use sigmoid not softmax
    import torch.nn as nn
    print(out)
    print(nn.Sigmoid()(out))

    # Or binary classification * N (for Grad-CAM)
    seresnet = timm.create_model('seresnet50', True, num_classes=num_classes*2)
    x = torch.randn((2, 3, 512, 512))  # B, C, H, W
    out = seresnet(x)
    # break by 2 then softmax
    nn.Softmax()

    # then finetune ...
    # training code
