import torch
from auxiliary.my_utils import yellow_print
from model.model import EncoderDecoder
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy

class TrainerModel(object):
    def __init__(self):
        """
        This class creates the architectures and implements all trainer functions related to architecture.
        Author : Thibault Groueix 01.11.2019
        """
        super(TrainerModel, self).__init__()

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if torch.cuda.is_available():
            self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")

        self.network = EncoderDecoder(self.opt)
        self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

        self.reload_network()

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.reload_model_path != "":
            yellow_print(f"Network weights loaded from  {self.opt.reload_model_path}!")
            # print(self.network.state_dict().keys())
            # print(torch.load(self.opt.reload_model_path).keys())
            self.network.module.load_state_dict(torch.load(self.opt.reload_model_path, map_location='cuda:0', weights_only=False))

        elif self.opt.reload_decoder_path != "":
            opt = deepcopy(self.opt)
            opt.SVR = False
            network = EncoderDecoder(opt)
            network = nn.DataParallel(network, device_ids=opt.multi_gpu)
            full = torch.load(opt.reload_decoder_path, map_location='cuda:0', weights_only=False)
            # Decoder-only transplant: the checkpoint's encoder (a resnet for
            # single-view checkpoints, a PointNet for autoencoder ones) need
            # not match ours — a strict full load would crash on it. Only
            # decoder.* tensors are taken, and anything less than the complete
            # decoder raises instead of silently partially loading (rule 4).
            own_sd = network.module.state_dict()
            dec = {k: v for k, v in full.items() if k.startswith("decoder.")}
            own_dec = [k for k in own_sd if k.startswith("decoder.")]
            missing = [k for k in own_dec if k not in dec or dec[k].shape != own_sd[k].shape]
            if not dec:
                raise RuntimeError(
                    f"[reload-decoder] no decoder.* tensors in {opt.reload_decoder_path}")
            if missing:
                raise RuntimeError(
                    f"[reload-decoder] {len(missing)} decoder tensors missing/mismatched "
                    f"(showing up to 8): {missing[:8]}")
            network.module.load_state_dict(dec, strict=False)
            self.network.module.decoder = network.module.decoder
            yellow_print(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path} "
                         f"({len(dec)}/{len(own_dec)} tensors)!")

        else:
            yellow_print("No network weights to reload!")

    def _trainable_params(self):
        """Parameters the optimizer may touch (honors --freeze_encoder)."""
        if getattr(self.opt, "freeze_encoder", False):
            return [p for n, p in self.network.named_parameters()
                    if ".encoder." not in n]
        return self.network.parameters()

    def build_optimizer(self):
        """
        Create optimizer
        """
        if self.opt.train_only_encoder:
            # To train a resnet image encoder with a pre-trained atlasnet decoder.
            yellow_print("only train the Encoder")
            self.optimizer = optim.Adam(self.network.module.encoder.parameters(), lr=self.opt.lrate)
        else:
            if getattr(self.opt, "freeze_encoder", False):
                for p in self.network.module.encoder.parameters():
                    p.requires_grad_(False)
                yellow_print("frozen image encoder; training decoder only")
            self.optimizer = optim.Adam(self._trainable_params(), lr=self.opt.lrate)

        if self.opt.reload_optimizer_path != "":
            try:
                self.optimizer.load_state_dict(torch.load(self.opt.reload_optimizer_path, map_location='cuda:0', weights_only=False))
                # yellow_print(f"Reloaded optimizer {self.opt.reload_optimizer_path}")
            except:
                yellow_print(f"Failed to reload optimizer {self.opt.reload_optimizer_path}")

        # Set policy for warm-up if you use multiple GPUs
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()
