import os
import os.path as osp
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import xmltodict
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

from .data_utils import nested_tensor_from_tensor_list

MAX_DIM = 256


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = transforms.Compose([
    RandomRotation(),
    transforms.Lambda(under_max),
    transforms.ColorJitter(brightness=[0.5, 1.3],
                           contrast=[0.8, 1.5],
                           saturation=[0.2, 1.5]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Lambda(under_max),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class Doc:
    def __init__(self, fd):
        self.parsed = xmltodict.parse(fd.read())

    def get_id(self):
        return self.parsed['eCitation']['uId']['@id']

    def get_findings(self):
        parsed_text = self.get_report_text()
        findings = parsed_text["FINDINGS"] if "FINDINGS" in parsed_text else ""
        return findings

    def get_report_text(self):
        abstract_text = self.parsed['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']
        dic_abstract_text = {}
        for val in abstract_text:
            if "#text" in val:
                dic_abstract_text[val['@Label']] = val['#text']
        return dic_abstract_text


class OpenIDataset(Dataset):

    def __init__(self, args, mode, transform=val_transform):
        super().__init__()
        self.transform = transform
        self.max_seq_length = args.max_position_embeddings + 1
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                       cache_dir=args.cache_dir if args.cache_dir else None)
        self.image_dir = os.path.join(args.data_dir, 'images')
        self.report_dir = os.path.join(args.data_dir, 'reports')
        self.label_dir = osp.join(args.data_dir, 'labels')
        with open(osp.join(args.data_dir, f'{mode}_img_names.txt'), 'r') as f:
            self.img_names = f.readlines()
        self.img_names = [i[:-1] for i in self.img_names]

    def get_image(self, img_name):
        image_full_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_full_path)
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        return image

    def get_label(self, img_name):
        report_id = img_name.split('_')[0][3:]  # CXR<num>_... -> <num>
        label_name = f'{report_id}_labeled.csv'
        label_path = osp.join(self.label_dir, label_name)
        df = pd.read_csv(label_path).fillna(0)
        # assert self.columns == list(df.columns)
        array = df.values[:, 2:]
        label = np.sum(array, axis=0)
        label = np.concatenate([[float((label == 0).all())], label])
        label = label.astype(np.float)
        label = torch.from_numpy(label).float()
        label[label > 0] = 1
        label[label < 0] = 0.5
        return label

    def get_report(self, img_name):
        report_id = img_name.split('_')[0][3:]  # CXR<num>_... -> <num>
        with open(os.path.join(self.report_dir, f"{report_id}.xml")) as fd:
            doc = Doc(fd)
        findings = doc.get_findings()
        caption_encoded = self.tokenizer.encode_plus(findings,
                                                     max_length=self.max_seq_length,
                                                     pad_to_max_length=True,
                                                     return_attention_mask=True,
                                                     return_token_type_ids=False,
                                                     truncation=True)
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return caption, cap_mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = self.get_image(self.img_names[idx])
        caption, cap_mask = self.get_report(self.img_names[idx])
        label = self.get_label(self.img_names[idx])
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask, label


class OpenIDataLoader:

    def __init__(self, args, batch_size):
        train_dataset = OpenIDataset(args, mode='train')
        val_dataset = OpenIDataset(args, mode='val')
        test_dataset = OpenIDataset(args, mode='test')

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
