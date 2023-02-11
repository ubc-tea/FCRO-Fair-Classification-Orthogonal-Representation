import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from utils.loss import ColOrthLoss, RowOrthLoss
from utils.util import check_utility, check_fairness


class FCROTrainer:
    def __init__(
        self,
        args,
        logger,
        mode,
        model_t=None,
        model_a=None,
        train_dataloader=None,
        val_dataloader=None,
    ):
        self.args = args
        self.mode = mode
        self.logger = logger
        self.model_t = model_t
        self.model_a = model_a

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.epoch = 0
        self.iteration = 0

        self.criterion = nn.BCEWithLogitsLoss()

        if self.mode == 1:
            if self.args.loss_row_weight:
                for param in self.model_a.parameters():
                    param.requires_grad = False

                self.row_criterion = RowOrthLoss(
                    conditional=self.args.cond,
                    margin=self.args.loss_row_margin,
                ).cuda()

            if self.args.loss_col_weight:
                if not self.args.moving_base:
                    U = self.generate_sensitive_subspace(
                        DataLoader(
                            self.train_dataloader.dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=8,
                        )
                    )
                else:
                    U = None

                self.col_criterion = ColOrthLoss(
                    U,
                    conditional=self.args.cond,
                    margin=self.args.loss_col_margin,
                    moving_base=self.args.moving_base,
                ).cuda()

            self.optimizer = torch.optim.Adam(
                self.model_t.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
            )

        elif self.mode == 0:
            self.optimizer = torch.optim.Adam(
                self.model_a.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.mode == -1:
            pass
        else:
            raise NotImplementedError

    def _update(self, minibatches):
        x, y, a = minibatches
        x, y = x.cuda(), y.cuda()

        loss_dict = {}
        loss = 0.0

        if self.mode == 0:
            self.model_a.train()

            out, emb = self.model_a(x)

            loss_sa = []
            for i, sa in enumerate(self.args.sensitive_attributes):
                loss_sa.append(
                    self.criterion(out[:, i].unsqueeze(1), a[sa].float().cuda())
                    / len(self.args.sensitive_attributes)
                )
                loss = loss + loss_sa[i]
                loss_dict[f"loss_sa_{sa}"] = loss_sa[i].item()

        elif self.mode == 1:
            self.model_a.eval()
            self.model_t.train()

            out, emb = self.model_t(x)
            with torch.no_grad():
                out_a, emb_a = self.model_a(x)

            loss_sup = self.criterion(out, y.float())
            loss_dict["loss_sup"] = loss_sup.item()
            loss = loss + loss_sup

            if self.args.loss_col_weight:
                if self.args.moving_base:
                    loss_col = (
                        self.col_criterion(emb, y, emb_a, self.epoch) * self.args.loss_col_weight
                    )
                else:
                    loss_col = self.col_criterion(emb, y) * self.args.loss_col_weight
                loss = loss + loss_col
                loss_dict["loss_col"] = loss_col.item()

            if self.args.loss_row_weight:
                loss_row = self.row_criterion(emb, emb_a.detach(), y) * self.args.loss_row_weight
                loss = loss + loss_row
                loss_dict["loss_row"] = loss_row.item()

        loss_dict["loss"] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1

        return loss_dict

    def train_epoch(self):
        for data in self.train_dataloader:
            loss = self._update(data)

            if self.iteration % self.args.log_step == 0:
                self.logger.info(
                    "[Epoch {}: Ite {}] {}".format(
                        self.epoch + 1,
                        self.iteration,
                        ", ".join("{}: {}".format(k, v) for k, v in loss.items()),
                    )
                )

        self.epoch += 1

    def validate_target(self):
        self.model_t.eval()

        targets = np.array([])
        a_dict = {}
        preds = np.array([])

        for sa in self.args.sensitive_attributes:
            a_dict[sa] = np.array([])

        with torch.no_grad():
            for input, target, z in self.val_dataloader:
                input = input.cuda()

                logits, _ = self.model_t(input)
                pred = torch.sigmoid(logits)

                targets = np.append(targets, target.squeeze().cpu().numpy())
                preds = np.append(preds, pred.squeeze().cpu().numpy())

                for i, sa in enumerate(self.args.sensitive_attributes):
                    a_dict[sa] = np.append(a_dict[sa], z[sa].squeeze().cpu().numpy())

        accuracy_dict = check_utility(preds, targets)
        fairness_metrics, subgroup_meta_data = check_fairness(preds, targets, a_dict)

        self.report_validation(accuracy_dict=accuracy_dict, fairness_metrics=fairness_metrics)

    def report_validation(
        self,
        accuracy_dict,
        fairness_metrics,
    ):
        base = "[Validating target head on epoch {}]\n".format(self.epoch)
        base += "<Utility> - {}\n".format(
            ", ".join("{}: {}".format(k, v) for k, v in accuracy_dict.items())
        )
        base += "<Joint Fairness> - {}\n".format(
            ", ".join("{}: {}".format(k, v) for k, v in fairness_metrics["combination"].items())
        )
        for sa in self.args.sensitive_attributes:
            base += "<Individual Fairness on {}> - {}\n".format(
                sa,
                ", ".join(
                    "{}: {}".format(k, v) for k, v in fairness_metrics["single"][sa].items()
                ),
            )
        self.logger.info(base)

    def validate_sensitive(self):
        self.model_a.eval()

        a_dict = {}
        pred_dict = {}

        for sa in self.args.sensitive_attributes:
            a_dict[sa] = np.array([])
            pred_dict[sa] = np.array([])

        with torch.no_grad():
            for input, _, z in self.val_dataloader:
                input = input.cuda()

                logits, _ = self.model_a(input)
                logits = torch.sigmoid(logits)

                for i, sa in enumerate(self.args.sensitive_attributes):
                    a_dict[sa] = np.append(a_dict[sa], z[sa].squeeze().cpu().numpy())
                    pred_dict[sa] = np.append(pred_dict[sa], logits[:, i].squeeze().cpu().numpy())

        auc_dict = {}
        for sa in self.args.sensitive_attributes:
            auc_dict[f"AUC_{sa}"] = roc_auc_score(a_dict[sa], pred_dict[sa])

        self.logger.info(
            "[Validating sensitive head on epoch {}] {}".format(
                self.epoch, ", ".join("{}: {}".format(k, v) for k, v in auc_dict.items())
            )
        )

    def run(self):
        self.logger.info(f"-- Start training mode {self.mode} of fold {self.args.fold}.")
        for _ in range(self.args.epoch):
            self.train_epoch()

            if self.mode == 0:
                self.validate_sensitive()
                self.save(os.path.join(self.args.output_dir, f"model_sensitive_latest.pth"))
            elif self.mode == 1:
                self.validate_target()
                self.save(os.path.join(self.args.output_dir, f"model_target_{self.epoch}.pth"))
            else:
                raise NotImplementedError

        self.logger.info(f"-- Finish training mode {self.mode}.")

    def generate_sensitive_subspace(self, dataloader):
        assert self.mode == 1, "Subspace is needed only when training target head."

        self.logger.info(
            f"Building static subspace for sensitive representations on {len(dataloader.dataset)} samples."
        )

        emb = []
        targets = []
        with torch.no_grad():
            for input, target, _ in dataloader:
                input = input.cuda()
                emb.append(self.model_a(input)[1])
                targets.append(target)

        emb = torch.concat(emb, dim=0).cpu()
        targets = torch.concat(targets, dim=0).squeeze().cpu()

        U_list = []
        for i in range(int(self.args.cond) + 1):
            if self.args.cond:
                indices = torch.where(targets == i)[0]
                emb_sub = torch.index_select(emb, 0, indices)
            else:
                emb_sub = emb

            U, S, _ = torch.linalg.svd(emb_sub.T, full_matrices=False)

            sval_ratio = (S**2) / (S**2).sum()
            r = (torch.cumsum(sval_ratio, -1) <= self.args.subspace_thre).sum()
            U_list.append(U[:, :r])

        return U_list

    def get_models(self):
        return self.model_t, self.model_a

    def save(self, path):
        net = self.model_t if self.mode == 1 else self.model_a

        if isinstance(net, nn.DataParallel):
            torch.save({"state_dict": net.module.state_dict()}, path)
        else:
            torch.save({"state_dict": net.state_dict()}, path)
