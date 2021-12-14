import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import logging
import math
import torch
from tqdm import tqdm

from models import model_utils, caption
from datasets import openi
import utils

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args

        ### Load data ###
        loaders = openi.OpenIDataLoader(args, batch_size=args.batch_size)
        self.train_loader = loaders.train_loader
        self.val_loader = loaders.val_loader
        # self.test_loader = loaders.test_loader

        ### Load model ###
        self.model, self.criterion = caption.build_model(args)
        # if we use bert tokenizer, we can load COCO-pretrained model
        # self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        self.model.to(args.device)

        ### Setting optimizer and scheduler ###
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of parameters: {n_params}")

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.lr_drop)


    def train(self):
        best_val_loss = 10000
        for epoch in range(self.args.epochs):
            train_loss = self.__epoch_train()
            val_loss = self.__epoch_val()
            self.scheduler.step()

            logger.info(f"Epoch {epoch+1}: train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.scheduler.state_dict(),
                    'epoch': epoch+1,
                }, os.path.join(self.args.output_dir, f"epoch{epoch+1}_{val_loss:.4f}.pt"))
            

    def __epoch_train(self):
        self.model.train()
        self.criterion.train()

        train_loss = 0.
        total = len(self.train_loader)

        with tqdm(total=total) as pbar:
            for images, masks, caps, cap_masks in self.train_loader:
                # samples.tensors (bs, 3, H, W) // samples.mask (bs, H, W)
                # caps, cap_masks (bs, max_seq_length)
                samples = model_utils.NestedTensor(images, masks).to(self.args.device)
                caps = caps.to(self.args.device)
                cap_masks = cap_masks.to(self.args.device)

                # outputs (bs, max_seq_length, vocab_size)
                outputs = self.model(samples, caps[:, :-1], cap_masks[:, :-1])
                loss = self.criterion(outputs.permute(0, 2, 1), caps[:, 1:])
                loss_value = loss.item()
                train_loss += loss_value

                if not math.isfinite(loss_value):
                    logger.warning(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                self.optimizer.zero_grad()
                loss.backward()
                if self.args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
                self.optimizer.step()

                pbar.update(1)
        
        return train_loss / total


    def __epoch_val(self):
        self.model.eval()
        self.criterion.eval()

        val_loss = 0.
        total = len(self.val_loader)

        with tqdm(total=total) as pbar:
            for images, masks, caps, cap_masks in self.val_loader:
                samples = model_utils.NestedTensor(images, masks).to(self.args.device)
                caps = caps.to(self.args.device)
                cap_masks = cap_masks.to(self.args.device)

                outputs = self.model(samples, caps[:, :-1], cap_masks[:, :-1])
                loss = self.criterion(outputs.permute(0, 2, 1), caps[:, 1:])

                val_loss += loss.item()

                pbar.update(1)

        return val_loss / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Model arguments ###
    parser.add_argument("--cache_dir", default=".cache/", type=str)

    # Backbone
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--position_embedding", type=str, default="sine")
    parser.add_argument("--dilation", action="store_false")

    # Transformer
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=31090)    # SciBERT vocab size
    parser.add_argument("--enc_layers", type=int, default=6)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--pre_norm", action="store_false")

    # ### Data arguments ###
    parser.add_argument("--data_dir", type=str, default="./data/openi")
    parser.add_argument("--data_limit", type=int, default=-1)

    # ### Training arguments ###
    parser.add_argument("--output_dir", default=".checkpoints/openi/", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_drop", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_max_norm", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Device: {args.device}, n_gpu: {args.n_gpu}")

    # Set seed
    logger.info(f"Setting seed {args.seed}")
    utils.set_seed(args.seed)

    trainer = Trainer(args)
    trainer.train()
