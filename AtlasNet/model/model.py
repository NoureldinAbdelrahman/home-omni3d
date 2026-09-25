from model.atlasnet import Atlasnet
from model.model_blocks import PointNet
import torch
import torch.nn as nn
import model.resnet as resnet

# ImageNet input statistics for the pretrained ResNet-18 encoder.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        self.svr = bool(opt.SVR)
        if self.svr:
            # ImageNet-pretrained encoder (fc head rebuilt to bottleneck_size).
            self.encoder = resnet.resnet18(pretrained=True, num_classes=opt.bottleneck_size)
            # Registered as buffers so training, evaluation and generated
            # meshes all apply the same normalization, and it travels inside
            # the saved checkpoint.
            self.register_buffer(
                "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer(
                "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        else:
            self.encoder = PointNet(nlatent=opt.bottleneck_size)

        self.decoder = Atlasnet(opt)
        self.to(opt.device)

        if not opt.SVR:
            self.apply(weights_init)  # initialization of the weights
        self.eval()

    def _normalize_image(self, x):
        return (x - self.img_mean) / self.img_std

    def forward(self, x, train=True):
        if self.svr and x.dim() == 4:  # single-view images [B,3,H,W]
            x = self._normalize_image(x)
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        if self.svr and x.dim() == 4:
            x = self._normalize_image(x)
        return self.decoder.generate_mesh(self.encoder(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
