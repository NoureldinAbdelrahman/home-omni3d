from torch import nn
from torch.autograd import Function
import torch
import importlib.util
import os

chamfer_found = importlib.util.find_spec("chamfer_3D") is not None
USE_CUDA_EXT = False
if chamfer_found:
    import chamfer_3D
    USE_CUDA_EXT = True
    print("Loaded compiled 3D CUDA chamfer distance")
else:
    try:
        print("Jitting Chamfer 3D")
        from torch.utils.cpp_extension import load
        this_dir = os.path.dirname(os.path.abspath(__file__))
        chamfer_3D = load(
            name="chamfer_3D",
            sources=[
                os.path.join(this_dir, "chamfer_cuda.cpp"),
                os.path.join(this_dir, "chamfer3D.cu"),
            ],
        )
        USE_CUDA_EXT = True
        print("Loaded JIT 3D CUDA chamfer distance")
    except Exception as e:
        print(f"CUDA chamfer unavailable ({e}); using pure-PyTorch fallback")


def _pairwise_min(x, y):
    """Nearest-neighbor squared distances from each point in x to points in y.
    x: (B, N, 3), y: (B, M, 3) -> dist (B, N), idx (B, N) long
    """
    xx = torch.sum(x ** 2, dim=2)
    yy = torch.sum(y ** 2, dim=2)
    zz = torch.bmm(x, y.transpose(1, 2))
    dist = xx.unsqueeze(2) + yy.unsqueeze(1) - 2.0 * zz
    dist = torch.clamp(dist, min=0.0)
    min_dist, min_idx = dist.min(dim=2)
    return min_dist, min_idx


class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if USE_CUDA_EXT and xyz1.is_cuda:
            batchsize, n, _ = xyz1.size()
            _, m, _ = xyz2.size()
            device = xyz1.device

            dist1 = torch.zeros(batchsize, n, device=device)
            dist2 = torch.zeros(batchsize, m, device=device)
            idx1 = torch.zeros(batchsize, n, dtype=torch.int32, device=device)
            idx2 = torch.zeros(batchsize, m, dtype=torch.int32, device=device)

            torch.cuda.set_device(device)
            chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
            ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
            ctx.use_cuda = True
            return dist1, dist2, idx1, idx2

        dist1, idx1 = _pairwise_min(xyz1, xyz2)
        dist2, idx2 = _pairwise_min(xyz2, xyz1)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.use_cuda = False
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        if getattr(ctx, "use_cuda", False):
            graddist1 = graddist1.contiguous()
            graddist2 = graddist2.contiguous()
            device = graddist1.device
            gradxyz1 = torch.zeros(xyz1.size(), device=device)
            gradxyz2 = torch.zeros(xyz2.size(), device=device)
            chamfer_3D.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )
            return gradxyz1, gradxyz2

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)
        # dist1_i = ||x_i - y_nn||^2  =>  d dist/d x = 2 (x - y_nn)
        if graddist1 is not None:
            x_nn = torch.gather(xyz2, 1, idx1.unsqueeze(2).expand(-1, -1, 3))
            gradxyz1 = gradxyz1 + 2.0 * (xyz1 - x_nn) * graddist1.unsqueeze(2)
        if graddist2 is not None:
            y_nn = torch.gather(xyz1, 1, idx2.unsqueeze(2).expand(-1, -1, 3))
            gradxyz2 = gradxyz2 + 2.0 * (xyz2 - y_nn) * graddist2.unsqueeze(2)
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)
