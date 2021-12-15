import os
import pickle
import random
import numpy as np
import SimpleITK as sitk

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .data_utils import nested_tensor_from_tensor_list

MAX_DIM = 299


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = transforms.Compose([
    RandomRotation(),
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=[0.5, 1.3],
                           contrast=[0.8, 1.5],
                           saturation=[0.2, 1.5]),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(2549.440547823737, 2549.440547823737, 2549.440547823737),
                         std=(1490.4394489321774, 1490.4394489321774, 1490.4394489321774)),
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=(2549.440547823737, 2549.440547823737, 2549.440547823737),
                         std=(1490.4394489321774, 1490.4394489321774, 1490.4394489321774)),
])


class MIMICCXRDataset(Dataset):

    def __init__(self, args, transform=train_transform, split='train'):
        super().__init__()
        self.transform = transform
        self.max_seq_length = args.max_position_embeddings + 1
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                        cache_dir=args.cache_dir if args.cache_dir else None)

        ## 파일은 완료되면 nas로 이동 예정
        if split == 'train':
            with open('/home/nas1_userE/gyuhyeonsim/train_file_list.pickle', 'rb') as f:
                self.image_paths = pickle.load(f)['file_list']
        elif split == 'valid':
            with open('/home/nas1_userE/gyuhyeonsim/valid_file_list.pickle', 'rb') as f:
                self.image_paths = pickle.load(f)['file_list']
        elif split == 'test':
            with open('/home/nas1_userE/gyuhyeonsim/test_file_list.pickle', 'rb') as f:
                self.image_paths = pickle.load(f)['file_list']

        
    def get_report_path_from_image_path(self, image_path):
        file_meta_info = image_path.split('/')
        base_path = "/".join(file_meta_info[:-3])
        patient_id = file_meta_info[-3]
        study_id = file_meta_info[-2]
        file_id = file_meta_info[-1]
        return os.path.join(base_path, patient_id, study_id + ".txt")

    def get_image(self, image_path):
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image).astype(np.float)  # to numpy
        image = torch.from_numpy(image)
        image = torch.cat([image, image, image], dim=0).unsqueeze(0)
        image /= image.max()
        image = image.float()
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image)
        return image

    def get_report(self, report_path):
        with open(report_path, 'r') as f:
            text = f.read()
            findings = text.split("FINDINGS:")[1].split("IMPRESSION:")[0].strip().replace("\n", "").replace("  ", " ")
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.get_image(image_path)
        report_path = self.get_report_path_from_image_path(image_path)
        caption, cap_mask = self.get_report(report_path)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


class MIMICCXRDataLoader:

    def __init__(self, args, batch_size):
        train_dataset = MIMICCXRDataset(args, transform=train_transform, split='train')
        valid_dataset = MIMICCXRDataset(args, transform=valid_transform, split='valid')
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
