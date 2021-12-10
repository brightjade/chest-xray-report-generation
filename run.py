import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

import dataset
import encoder
import decoder
from models import *

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.transform = transforms.Compose([transforms.Resize(args.resize),
                                             transforms.RandomCrop(args.crop_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])

        loaders = dataset.CXRDataLoader(args.data_dir, batch_size=args.batch_size, transform=self.transform)
        dic = dataset.dic
        print(len(dic))

        self.train_loader = loaders.train_loader
        self.val_loader = loaders.val_loader
        self.test_loader = loaders.test_loader

        # with open(args.vocab_path, "rb") as f:
        #     self.vocab = pickle.load(f)

        self.encoderCNN = encoder.EncoderCNN().to(args.device)     # TODO: replace later
        self.sentenceRNN = decoder.SentenceRNN().to(args.device)
        self.wordRNN = decoder.WordRNN(hidden_size=256, vocab_size=len(dic), att_dim=256, embed_size=256, encoded_dim=256, device=args.device).to(args.device)

        self.criterion_sentence = nn.BCELoss(size_average=False, reduce=False).to(args.device)
        self.criterion_word = nn.CrossEntropyLoss().to(args.device)

        params = list(self.encoderCNN.parameters()) + list(self.sentenceRNN.parameters()) + list(self.wordRNN.parameters())
        self.optimizer = optim.Adam(params=params, lr=self.args.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)


    def train(self):
        for epoch in range(self.args.epochs):
            train_loss = self.__epoch_train()
            # val_loss = self.__epoch_val()
            val_loss = 0
            self.scheduler.step(train_loss)
            logger.info(f"Epoch {epoch+1}: train loss: {train_loss}, val_loss: {val_loss}")
            # TODO: save best model
            

    def __epoch_train(self):
        train_loss = 0
        self.encoderCNN.train()
        self.sentenceRNN.train()
        self.wordRNN.train()

        for i, (images, findings, sent_vectors, word_vectors, word_lengths) in enumerate(tqdm(self.train_loader, desc="Training")):
            images = images.to(self.args.device)
            word_vectors = word_vectors.to(self.args.device)
            feature_map, global_features = self.encoderCNN(images)  # feature_map = (bs, 49, 256)
                                                                    # global_features = (bs, 256)
            sent_states = None
            loss, sent_loss, word_loss = 0, 0, 0
            word_vectors = word_vectors.permute(1, 0, 2)    # (bs, 15, 55) -> (15, bs, 55)

            for sent_idx, sent_value in enumerate(sent_vectors):
                # topic_vec = (bs, 1, 256) // sent_states (hn, cn) = (1, bs, 256)
                end_token, topic_vec, sent_states = self.sentenceRNN(global_features, sent_states)
                end_token = end_token.squeeze(1).squeeze(1)     # end_token = (bs, 1, 1) -> (bs,)

                # Loss for # of generated sentences; sent_value = [1, 1, 1, ..., 0, 0, ...]
                sent_loss = sent_loss + self.criterion_sentence(end_token, sent_value.type(torch.float).to(self.args.device)).sum()

                captions = word_vectors[sent_idx]           # ground-truth ith sentence = (bs, 55)
                caption_lengths = word_lengths[sent_idx]    # ground-truth ith sentence length = (bs,)

                if any(caption_lengths):
                    # predictions = (bs, max(decode_lengths), vocab_size) //
                    # alphas = (bs, max(decode_lengths), num_pixels+1) // betas = (bs, max(decode_lengths), 1)
                    # enc_captions = (bs, 55) // decode_lengths = (bs,) // sort_ind = (bs,)
                    predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = self.wordRNN(
                                                                                                enc_image=feature_map,
                                                                                                global_features=global_features,
                                                                                                encoded_captions=captions,
                                                                                                caption_lengths=caption_lengths)
                    targets = captions

                    greaterThan0LengthIndeces = list() #remove length 0 sentences
                    greaterThan0Lengths = list()
                    for i, length in enumerate(decode_lengths):
                        if(length > 0):
                            greaterThan0LengthIndeces.append(i)
                            greaterThan0Lengths.append(length)

                    targets = targets[greaterThan0LengthIndeces]            # (bs, 55)
                    predictions = predictions[greaterThan0LengthIndeces]    # (bs, max(decode_lengths), vocab_size)

                    # Remove all unnecessary paddings at the end
                    targets = pack_padded_sequence(targets, greaterThan0Lengths, batch_first=True).data
                    scores = pack_padded_sequence(predictions, greaterThan0Lengths, batch_first=True).data

                    word_loss = word_loss + self.criterion_word(scores, targets)

            loss = word_loss + sent_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            

        return train_loss


    def __epoch_val(self):
        val_loss = 0
        self.encoderCNN.eval()
        self.sentenceRNN.eval()
        self.wordRNN.eval()

        for i, (images, findings, sent_vectors, word_vectors, word_lengths) in enumerate(tqdm(self.val_loader, desc="Validating")):
            images = images.to(self.args.device)
            break

        return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Model arguments ###
    # CNN
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    # LSTM
    parser.add_argument("--P", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--word_lstm_layer", type=int, default=2)
    parser.add_argument("--sentence_lstm_layer", type=int, default=1)
    parser.add_argument("--pooling_dim", type=int, default=1024)
    parser.add_argument("--topic_dim", type=int, default=1024)

    ### Data arguments ###
    parser.add_argument("--data_dir", type=str, default="./data/")

    ### Training arguments ###
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Device: {args.device}, n_gpu: {args.n_gpu}")

    trainer = Trainer(args)
    trainer.train()
