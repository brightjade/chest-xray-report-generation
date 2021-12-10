import os
import re
import xmltodict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from tqdm import tqdm

MAX_NUMBER_OF_SENTENCES=15
MAX_SENTENCE_LENGTH=55


class Dictionary:
    EOS = '<eos>'
    SOS = '<sos>'

    def __init__(self):
        self.word_id = {}
        self.id_word = list()
        self.next_id = 0
        self.add_word(Dictionary.EOS)
        self.add_word(Dictionary.SOS)
    
    def add_word(self, word):
        cw = re.sub("\.$|\,$", "", word).lower()
        if cw not in self.word_id:
            self.word_id[cw] = self.next_id
            self.id_word.append(cw)
            self.next_id = self.next_id + 1
        return cw, self.word_id[cw]

    def get_word_id(self, word):
        cw, _id = self.add_word(word)
        return _id

    def __len__(self):
        return self.next_id

dic = Dictionary()

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


class Sentences:
    def __init__(self, findings):
        self.sentences = re.split('\. ', findings or '')
    
    def sentences_vector(self):
        vector = [0]*MAX_NUMBER_OF_SENTENCES
        for i in range(len(self)):
            vector[i] = 1
        return vector

    def __len__(self):
        return len(self.sentences)

    def init_word_vector(self):
        return [[dic.get_word_id(Dictionary.EOS)]*MAX_SENTENCE_LENGTH for _ in range(MAX_NUMBER_OF_SENTENCES)]

    def word_vector(self):
        error = False
        word_vector = self.init_word_vector()
        word_lengths = [0]*MAX_NUMBER_OF_SENTENCES
        for i, sentence in enumerate(self.sentences):
            words = sentence.split(' ')
            if len(words) == 0 or len(words) > MAX_SENTENCE_LENGTH:
                error = True
            for j in range(len(words)):
                word_vector[i][j] = dic.get_word_id(words[j])
            word_lengths[i] = len(words)
        return word_vector, word_lengths, error

    def init_word_lengths_vector(self):
        word_vector_lengths = [0]*MAX_NUMBER_OF_SENTENCES


class CXRDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.report_dir = os.path.join(data_dir, 'ecgen-radiology')
        self.image_paths = self.get_image_paths()
        self.report_paths = os.listdir(self.report_dir)
        self.skipped_images = set()
        self.reports = {}
        self.get_reports()
        self.clean_image_paths()    # exclude images without report

        print(f"# skipped images: {len(self.skipped_images)}")
        print(f"# filtered images: {len(self.image_paths)}")


    def get_image_paths(self):
        return list(filter(lambda p: p[-3:]=="png", os.listdir(self.image_dir)))

    def get_reports(self):
        for path in tqdm(self.report_paths, desc="Making reports to features"):
            with open(os.path.join(self.report_dir, path)) as fd:
                doc = Doc(fd)
                findings = doc.get_findings()   # text string
                sentences = Sentences(findings)

                if findings == "":
                    self.skipped_images.add(doc.get_id())
                    continue

                # sentence_vector = [1, 1, 1, 1, 1, 0, 0, ...] (1 if sentence exists else 0)
                if len(sentences) < MAX_NUMBER_OF_SENTENCES and len(sentences) > 0:
                    sentence_vector = sentences.sentences_vector()
                else:
                    self.skipped_images.add(doc.get_id())
                    continue

                # word_vector = [[2, 3, 4, 5, 0, 0, ...], ...] (basically, input_ids)
                # word_lengths = [10, 5, 5, ...] (word lengths for each sentence)
                word_vector, word_lengths, error = sentences.word_vector()

                if error:
                    self.skipped_images.add(doc.get_id())
                    continue

                self.reports[doc.get_id()] = {
                    "findings": findings,
                    "word_vector": word_vector,
                    "sent_vector": sentence_vector,
                    "word_lengths": word_lengths
                }

    def get_image(self, image_path):
        image_full_path = os.path.join(self.image_dir, image_path)
        image = Image.open(image_full_path)
        if self.transform:
            image = self.transform(image)
        return image

    def get_report_id_from_image_path(self, image_path):
        return image_path.split('_')[0]

    def clean_image_paths(self):
        temp_image_paths = []
        for image_name in self.image_paths:
            report_id = image_name.split('_')[0]
            if not (report_id in self.skipped_images):
                temp_image_paths.append(image_name)
        self.image_paths = temp_image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.get_image(image_path)
        report_id = self.get_report_id_from_image_path(image_path)
        report = self.reports[report_id]
        return image, report["findings"], report["sent_vector"], torch.tensor(report["word_vector"]), report["word_lengths"]


class CXRDataLoader:

    def __init__(self, data_dir, batch_size, transform=None):
        dataset = CXRDataset(data_dir, transform=transform)
        train_size = int(len(dataset)*0.7)
        train_dataset, testval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        val_size = int(len(dataset)*0.2)
        val_dataset, test_dataset = random_split(testval_dataset, [val_size, len(dataset) - train_size - val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        

# For debugging...
if __name__ == "__main__":
    DATA_DIR = "./data/"
    BATCH_SIZE = 16
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loaders = CXRDataLoader(DATA_DIR, BATCH_SIZE, TRANSFORM)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader

    print(f"Size of train: {len(train_loader) * BATCH_SIZE}")
    print(f"Size of val: {len(val_loader) * BATCH_SIZE}")
    print(f"Size of test: {len(test_loader)}")
