import os
import random
import xmltodict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer

from .data_utils import nested_tensor_from_tensor_list

MAX_DIM = 299


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

    def __init__(self, args, transform=val_transform):
        super().__init__()
        self.transform = transform
        self.max_seq_length = args.max_position_embeddings + 1
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                        cache_dir=args.cache_dir if args.cache_dir else None)
        self.image_dir = os.path.join(args.data_dir, 'images')
        self.report_dir = os.path.join(args.data_dir, 'reports')
        self.image_paths = self.get_image_paths()
        self.report_paths = os.listdir(self.report_dir)
        self.skipped_images = set()
        self.filter_reports()
        self.clean_image_paths()    # exclude images without report

        print(f"# skipped images: {len(self.skipped_images)}")
        print(f"# filtered images: {len(self.image_paths)}")

    def get_image_paths(self):
        return list(filter(lambda p: p[-3:]=="png", os.listdir(self.image_dir)))

    def get_report_id_from_image_path(self, image_path):
        return image_path.split('_')[0]

    def filter_reports(self):
        for path in tqdm(self.report_paths, desc="Filtering empty reports"):
            with open(os.path.join(self.report_dir, path)) as fd:
                doc = Doc(fd)
                findings = doc.get_findings()
                if findings == "":
                    self.skipped_images.add(doc.get_id())

    def clean_image_paths(self):
        temp_image_paths = []
        for image_name in self.image_paths:
            report_id = image_name.split('_')[0]
            if not (report_id in self.skipped_images):
                temp_image_paths.append(image_name)
        self.image_paths = temp_image_paths

    def get_image(self, image_path):
        image_full_path = os.path.join(self.image_dir, image_path)
        image = Image.open(image_full_path)
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        return image

    def get_report(self, report_id):
        # CXR<num> -> <num>
        with open(os.path.join(self.report_dir, report_id[3:]+".xml")) as fd:
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.get_image(image_path)
        report_id = self.get_report_id_from_image_path(image_path)
        caption, cap_mask = self.get_report(report_id)
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


class OpenIDataLoader:

    def __init__(self, args, batch_size):
        dataset = OpenIDataset(args)
        train_size = int(len(dataset)*0.7)
        train_dataset, testval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        val_size = int(len(dataset)*0.2)
        val_dataset, test_dataset = random_split(testval_dataset, [val_size, len(dataset) - train_size - val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
