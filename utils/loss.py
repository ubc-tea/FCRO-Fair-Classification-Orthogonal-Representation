import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RowOrthLoss(nn.Module):
    def __init__(self, conditional=False, margin=0.0):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super().__init__()

        ####
        self.conditional = conditional
        self.margin = margin

    def forward(self, feature_t, feature_a, class_label):
        # Apply gradient reversal on input embeddings.
        if self.conditional:
            feature_t_0, feature_t_1, feature_s_0, feature_s_1 = self.split(
                feature_t, feature_a, class_label
            )
            t_list = [feature_t_0, feature_t_1]
            s_list = [feature_s_0, feature_s_1]
        else:
            t_list = [feature_t]
            s_list = [feature_a]

        loss = 0.0
        for i, (feature_t_sub, feature_a_sub) in enumerate(zip(t_list, s_list)):
            feature_t_sub = feature_t_sub - feature_t_sub.mean(0, keepdim=True)
            feature_a_sub = feature_a_sub - feature_a_sub.mean(0, keepdim=True)

            sigma = torch.matmul(feature_t_sub.T, feature_a_sub.detach())

            sigma_loss = torch.clamp(torch.sum(sigma**2) - self.margin, min=0)
            loss = loss + sigma_loss / sigma.numel()

        loss = loss / len(t_list)

        return loss

    def split(self, feature_t, feature_s, class_label):
        indices1 = torch.where(class_label == 1)[0]
        indices0 = torch.where(class_label == 0)[0]

        feature_t_0 = torch.index_select(feature_t, 0, indices0)
        feature_t_1 = torch.index_select(feature_t, 0, indices1)

        feature_s_0 = torch.index_select(feature_s, 0, indices0)
        feature_s_1 = torch.index_select(feature_s, 0, indices1)

        return feature_t_0, feature_t_1, feature_s_0, feature_s_1


class ColOrthLoss(nn.Module):
    def __init__(
        self,
        U=None,
        conditional=False,
        margin=0.0,
        moving_base=False,
        threshold=0.99,
        moving_epoch=3,
    ):
        super(ColOrthLoss, self).__init__()
        self.conditional = conditional
        self.margin = margin
        self.moving_base = moving_base
        self.threshold = threshold
        self.moving_epoch = moving_epoch

        if not self.moving_base:
            self.U = U
        else:
            self.U = [None, None] if self.conditional else None

        if not isinstance(self.U, list):
            self.U = [self.U]

    def forward(self, feature_t, class_label, feature_a=None, epoch=None):
        if self.conditional:
            indices1 = torch.where(class_label == 1)[0]
            indices0 = torch.where(class_label == 0)[0]

            feature_t_0 = torch.index_select(feature_t, 0, indices0)
            feature_t_1 = torch.index_select(feature_t, 0, indices1)

            if self.moving_base:
                assert feature_a is not None
                feature_a_0 = torch.index_select(feature_a, 0, indices0)
                feature_a_1 = torch.index_select(feature_a, 0, indices1)

                a_list = [feature_a_0, feature_a_1]

            t_list = [feature_t_0, feature_t_1]
        else:
            t_list = [feature_t]
            if self.moving_base:
                a_list = [feature_a]

        loss = 0.0
        for i in range(len(t_list)):
            feature_t_sub = t_list[i]
            # feature_t_sub = feature_t_sub - feature_t_sub.mean(0, keepdim=True)

            if not self.moving_base:
                U_sub = self.U[i]
            else:
                feature_a_sub = a_list[i]
                if epoch is not None and epoch <= self.moving_epoch:
                    if self.U[i] is None:
                        U, S, _ = torch.linalg.svd(feature_a_sub.T, full_matrices=False)

                        sval_ratio = (S**2) / (S**2).sum()
                        r = (torch.cumsum(sval_ratio, -1) < self.threshold).sum()

                        self.U[i] = U[:, :r]
                    else:
                        with torch.no_grad():
                            self.update_space(feature_a_sub, condition=i)

                U_sub = self.U[i]

            proj_fea = torch.matmul(feature_t_sub, U_sub.to(feature_t.device))
            con_loss = torch.clamp(torch.sum(proj_fea**2) - self.margin, min=0)

            loss = loss + con_loss / feature_t_sub.shape[0]

        loss = loss / len(t_list)

        return loss

    def update_space(self, feature, condition):
        bases = self.U[condition].clone()

        R2 = torch.matmul(feature.T, feature)
        delta = []
        for ki in range(bases.shape[1]):
            base = bases[:, ki : ki + 1]
            delta_i = torch.matmul(torch.matmul(base.T, R2), base).squeeze()
            delta.append(delta_i)

        delta = torch.hstack(delta)

        _, S_, _ = torch.linalg.svd(feature.T, full_matrices=False)
        sval_total = (S_**2).sum()

        # projection_diff = feature - self.projection(feature, bases)
        projection_diff = feature - torch.matmul(torch.matmul(bases, bases.T), feature.T).T
        U, S, V = torch.linalg.svd(projection_diff.T, full_matrices=False)

        stack = torch.hstack((delta, S**2))
        S_new, sel_index = torch.topk(stack, len(stack))

        r = 0
        accumulated_sval = 0

        for i in range(len(stack)):
            if accumulated_sval < self.threshold * sval_total and r < feature.shape[1]:
                accumulated_sval += S_new[i].item()
                r += 1
            else:
                break

        sel_index = sel_index[:r]
        S_new = S_new[:r]

        Ui = torch.hstack([bases, U])
        U_new = torch.index_select(Ui, 1, sel_index)

        # sel_index_from_new = sel_index[sel_index >= len(delta)]
        # sel_index_from_old = sel_index[sel_index < len(delta)]
        # print(f"from old: {len(sel_index_from_old)}, from new: {len(sel_index_from_new)}")

        self.U[condition] = U_new.clone()
