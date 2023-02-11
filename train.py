import yaml
import random
import sys, os

import argparse
import time
import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy.random import shuffle

from dataset import CheXpert
from model.network import Network
from trainer.FCRO_trainer import FCROTrainer
import utils.util as util
from utils.data_sampler import ClassBalanceSampler


def main():
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay", help="weight decay of Adam optimizer", type=float, default=0.0004
    )
    parser.add_argument("--epoch", help="total training epochs", type=int, default=40)
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument(
        "--target_label",
        help="target classification disease",
        type=str,
        default="Pleural Effusion",
    )
    parser.add_argument(
        "-a",
        "--sensitive_attributes",
        help="set of sensitive attributes used for training and fairness testing",
        type=lambda s: [item for item in s.split(",")],
        default="Race,Sex,Age",
    )
    parser.add_argument(
        "--image_path", help="path of the CheXpert dataset", type=str, required=True
    )
    parser.add_argument(
        "--exp_path", help="path of the experiment result", type=str, default="../experiments"
    )
    parser.add_argument(
        "--pretrained_path",
        help="path of pretrained experiment path",
        type=str,
        default="./checkpoint",
    )
    parser.add_argument(
        "--metadata",
        help="path of the metadata file",
        type=str,
        default="./metadata/METADATA_resample_5_fold.xlsx",
    )
    parser.add_argument("--dim_rep", help="dimension of the representation", type=int, default=128)
    parser.add_argument("--cond", help="whether use conditional training", action="store_true")
    parser.add_argument(
        "-wc", "--loss_col_weight", help="weight of the coloumn loss", type=int, default=80
    )
    parser.add_argument(
        "-wr", "--loss_row_weight", help="weight of the row loss", type=int, default=500
    )
    parser.add_argument(
        "--loss_col_margin", help="margin of the column loss", type=float, default=0.0
    )
    parser.add_argument(
        "--loss_row_margin", help="margin of the row loss", type=float, default=0.0
    )
    parser.add_argument(
        "--moving_base", help="using moving bases in column loss", action="store_true"
    )
    parser.add_argument(
        "--subspace_thre",
        help="the threshold for deciding how many bases will be kept when building sensitive subspaces",
        type=float,
        default=0.99,
    )
    parser.add_argument("--test", help="test the pretrained model", action="store_true")
    parser.add_argument(
        "--from_sketch",
        help="whether train sensitive head or use a pretrained one",
        action="store_true",
    )
    parser.add_argument(
        "-f", "--fold", help="fold index for 5-fold cross validation", type=int, default=0
    )
    parser.add_argument(
        "--log_step", help="interval in iterations for logging loss", type=int, default=50
    )
    args = parser.parse_args()

    # set the random seed so that the random permutations can be reproduced again
    SEED = args.fold
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args.output_dir = os.path.join(args.exp_path, f"fold_{args.fold}")
    util.mkdir(args.output_dir)

    model_t = Network(1, args=args, pretrained=True).cuda()
    model_a = Network(len(args.sensitive_attributes), args=args, pretrained=True).cuda()

    if args.test:
        logger = util.setup_logger(
            f"test_{args.fold}", args.output_dir, f"test_{args.fold}", screen=True, tofile=True
        )

        test_set = CheXpert.CheXpertDataset(
            csv_path=args.metadata,
            image_root_path=args.image_path,
            target_labels=args.target_label,
            sensitive_attribute=args.sensitive_attributes,
            transform=CheXpert.test_transform,
            shuffle=False,
            mode=f"test",
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )

        model_t.load_state_dict(
            torch.load(os.path.join(args.pretrained_path, f"fold{args.fold}", "model_target.pth"))[
                "state_dict"
            ]
        )
        trainer = FCROTrainer(args, logger, mode=-1, model_t=model_t, val_dataloader=test_loader)
        logger.info(f"Starting test on {len(test_set)} samples of fold {args.fold}.")
        trainer.validate_target()

        exit(0)

    logger = util.setup_logger(
        f"train_{args.fold}_{util.get_timestamp()}",
        args.output_dir,
        f"train_{args.fold}",
        screen=True,
        tofile=True,
    )

    train_set = CheXpert.CheXpertDataset(
        csv_path=args.metadata,
        image_root_path=args.image_path,
        target_labels=args.target_label,
        sensitive_attribute=args.sensitive_attributes,
        transform=CheXpert.train_transform,
        shuffle=True,
        mode="train",
        sheet=f"train_{args.fold}",
    )

    val_set = CheXpert.CheXpertDataset(
        csv_path=args.metadata,
        image_root_path=args.image_path,
        target_labels=args.target_label,
        sensitive_attribute=args.sensitive_attributes,
        transform=CheXpert.test_transform,
        shuffle=False,
        mode=f"val",
        sheet=f"val_{args.fold}",
    )

    if args.cond:
        sampler = ClassBalanceSampler(
            y=train_set.targets, batch_size=args.batch_size, drop_last=True
        )
        train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=8)
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    if args.from_sketch:
        train_sensitive_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8
        )

        trainer_a = FCROTrainer(
            args,
            logger,
            mode=0,
            model_t=model_t,
            model_a=model_a,
            train_dataloader=train_sensitive_loader,
            val_dataloader=val_loader,
        )

        trainer_a.run()
        model_t, model_a = trainer_a.get_models()
    else:
        checkpoint_file = os.path.join(
            args.pretrained_path, f"fold{args.fold}", "model_sensitive.pth"
        )
        model_a.load_state_dict(torch.load(checkpoint_file)["state_dict"])

    trainer = FCROTrainer(
        args,
        logger,
        mode=1,
        model_t=model_t,
        model_a=model_a,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    trainer.run()


if __name__ == "__main__":
    main()
