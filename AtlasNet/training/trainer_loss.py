import torch
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D


from auxiliary.ChamferDistancePytorch.fscore import fscore
import os
import training.metro as metro
from joblib import Parallel, delayed
import numpy as np


class TrainerLoss(object):
    """
    This class implements all functions related to the loss of Atlasnet, mainly applies chamfer and metro.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        super(TrainerLoss, self).__init__()

    def build_losses(self):
        """
        Create loss functions.
        """
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.loss_model = self.chamfer_loss

    def fuse_primitives(self):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer
        Input size : batch, prim, 3, npoints
        Output size : prim, prim*npoints, 3
        :return:
        """
        #
        self.data.pointsReconstructed = self.data.pointsReconstructed_prims.transpose(2, 3).contiguous()
        self.data.pointsReconstructed = self.data.pointsReconstructed.view(self.batch_size, -1, 3)

    def chamfer_loss(self):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        inCham1 = self.data.points.view(self.data.points.size(0), -1, 3).contiguous()
        inCham2 = self.data.pointsReconstructed.contiguous().view(self.data.points.size(0), -1, 3).contiguous()

        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        self.data.loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        if not self.flags.train:
            self.data.loss_fscore, _, _ = fscore(dist1, dist2)
            self.data.loss_fscore = self.data.loss_fscore.mean()

    def patch_smoothness_loss(self, k=4):
        """Mean squared distance to the k nearest intra-patch neighbors.

        Penalizes stretched/spiky patches (a failure mode plain Chamfer
        forgives, since spikes contribute little to the mean) without needing
        GT normals. Works for any primitive count / template / point count.
        prims arrive as [B, prim, 3, P] (see fuse_primitives).
        """
        prims = self.data.pointsReconstructed_prims.transpose(2, 3).contiguous()
        B, M, P, _ = prims.shape
        d = torch.cdist(prims, prims)  # [B, M, P, P]
        d.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        nn_idx = d.topk(min(k, P - 1), dim=-1, largest=False).indices
        kk = nn_idx.shape[-1]
        # src[b,m,s,t,:] must equal prims[b,m,t,:]: unsqueeze the KEY dim
        # (dim 2), not the query dim. NOTE: .contiguous() is load-bearing —
        # torch.gather silently ignores the index on stride-0 (expanded)
        # dims, returning broadcast copies (both verified numerically).
        src = prims.unsqueeze(2).expand(B, M, P, P, 3).contiguous()
        neigh = torch.gather(src, dim=3,
                             index=nn_idx.unsqueeze(-1).expand(B, M, P, kk, 3))
        return ((prims.unsqueeze(3) - neigh).pow(2).sum(-1)).mean()

    def metro(self):
        """
        Compute the metro distance on a randomly selected test files.
        Uses joblib to leverage as much cpu as possible
        :return:
        """
        metro_path = './dataset/data/metro_files'
        metro_files_path = '/'.join([metro_path, 'files-metro.txt'])
        self.metro_args_input = []
        if not os.path.exists(metro_files_path):
            os.system("chmod +x dataset/download_metro_files.sh")
            os.system("./dataset/download_metro_files.sh")
        ext = '.png' if self.opt.SVR else '.npy'
        with open(metro_files_path, 'r') as file:
            files = file.read().split('\n')

        for file in files:
            if file[-3:] == "ply":
                cat = file.split('/')[0]
                name = file.split('/')[1][:-4]
                input_path = '/'.join([metro_path, cat, name + ext])
                input_path_points = '/'.join([metro_path, cat, name + '.npy'])
                gt_path = '/'.join([metro_path, cat, name + '.ply'])
                path = self.demo(input_path, input_path_points)
                self.metro_args_input.append((path, gt_path))

        print("start metro calculus. This is going to take some time (30 minutes)")
        self.metro_results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(metro.metro)(*i) for i in self.metro_args_input)
        self.metro_results = np.array(self.metro_results).mean()
        print(f"Metro distance : {self.metro_results}")
