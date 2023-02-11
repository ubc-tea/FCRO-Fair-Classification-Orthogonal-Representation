import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd


train_transform = tfs.Compose(
    [
        tfs.Resize([224, 224]),
        tfs.RandomRotation(degrees=(0, 180)),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = tfs.Compose(
    [
        tfs.Resize([224, 224]),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root_path,
        target_labels=None,
        sensitive_attribute="Race",
        shuffle=True,
        transform=None,
        mode="train",
        sheet=None,
    ):

        if target_labels is None:
            target_labels = [
                "Atelectasis",
                "Cardiomegaly",
                "Consolidation",
                "Edema",
                "Pleural Effusion",
            ]

        self.image_root_path = image_root_path
        self.target_labels = target_labels if isinstance(target_labels, list) else [target_labels]
        self.sensitive_attributes = sensitive_attribute
        self.transform = transform
        self.mode = mode

        if isinstance(self.sensitive_attributes, str):
            self.sensitive_attributes = [self.sensitive_attributes]

        if sheet is None:
            self.df = pd.read_excel(csv_path, sheet_name=self.mode)
        else:
            self.df = pd.read_excel(csv_path, sheet_name=sheet)

        # impute missing values
        for col in self.target_labels:
            if col in ["Edema", "Atelectasis"]:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ["Cardiomegaly", "Consolidation", "Pleural Effusion"]:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in [
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Lung Opacity",
                "Lung Lesion",
                "Pneumonia",
                "Pneumothorax",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ]:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self.num_imgs = len(self.df)

        for sa in self.sensitive_attributes:
            if sa == "Race":
                self.df[sa] = self.df[sa].apply(lambda x: 1 if "White" in x else 0)
            elif sa == "Sex":
                self.df[sa] = self.df[sa].apply(lambda x: 1 if x == "Male" else 0)
            elif sa == "Age":
                self.df[sa] = self.df[sa].apply(lambda x: 1 if x > 60 else 0)

        # shuffle data
        if shuffle:
            data_index = list(range(self.num_imgs))
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        self.images_list = [
            os.path.join(self.image_root_path, path) for path in self.df["Path"].tolist()
        ]
        self.targets = self.df[self.target_labels].values.tolist()
        self.a_dict = {}
        for attribute in self.sensitive_attributes:
            self.a_dict[attribute] = self.df[attribute].values.tolist()

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx]).view(-1).long()
        a = {}
        for k, v in self.a_dict.items():
            a[k] = torch.tensor(v[idx]).view(-1).long()

        return image, target, a
