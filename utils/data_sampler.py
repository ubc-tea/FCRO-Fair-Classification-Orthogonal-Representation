import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler


class ClassBalanceSampler(Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """

    def __init__(self, y, batch_size, drop_last=False):
        super(ClassBalanceSampler, self).__init__(y)
        self.y = np.array(y).astype(np.int16)
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            self.sampler_length = len(self.y) // self.batch_size
        else:
            self.sampler_length = (len(self.y) + self.batch_size - 1) // self.batch_size

        self.u_class_batch_num = self.batch_size // 2

        self.p_indices = np.where(self.y == 1)[0]
        self.u_indices = np.where(self.y == 0)[0]

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []

            subset.extend(
                np.random.choice(
                    self.p_indices, self.batch_size - self.u_class_batch_num, replace=False
                )
            )
            subset.extend(np.random.choice(self.u_indices, self.u_class_batch_num, replace=False))

            np.random.shuffle(subset)

            yield subset

    def __len__(self):
        return self.sampler_length
