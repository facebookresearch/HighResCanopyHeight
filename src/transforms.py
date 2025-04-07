import torch
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF


class Norm:
    def get_trans(self, nmean, nstd):
        self.nmean = nmean
        self.nstd = nstd
        self.Trans = transforms.Compose(
            [
                transforms.ToTensor(),
                # type: ignore
                transforms.Normalize(mean=self.nmean, std=self.nstd),
            ]
        )
        self.Normalize = transforms.Normalize(mean=self.nmean, std=self.nstd)
        self.invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1 / self.nstd[0], 1 / self.nstd[1], 1 / self.nstd[2]],
                ),
                transforms.Normalize(
                    mean=[-self.nmean[0], -self.nmean[1], -self.nmean[2]],
                    std=[1.0, 1.0, 1.0],
                ),
            ]
        )


class SSLNorm(Norm):
    def __init__(self):
        super().__init__()
        nmean = [0.430, 0.411, 0.296]
        nstd = [0.213, 0.156, 0.143]
        self.get_trans(nmean, nstd)
