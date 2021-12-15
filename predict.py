import os
import argparse
import logging
from PIL import Image
from tqdm import tqdm, trange

import torch
from transformers import AutoTokenizer

from models import caption
from datasets import openi

logger = logging.getLogger(__name__)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate(args, model, image, caption, cap_mask):
    model.eval()
    for i in trange(args.max_position_embeddings - 1, desc="Generating output"):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        # TODO: Beam Search
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == args.end_token:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption



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
    ### Data arguments ###
    parser.add_argument("--data_dir", type=str, default="./data/openi")
    parser.add_argument("--data_limit", type=int, default=-1)
    ### Training arguments ###
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

    print("Loading model")
    model, _ = caption.build_model(args)
    print("Loading checkpoint")
    checkpoint = torch.load(os.path.join(args.output_dir, "epoch18_0.4756.pt"))
    model.load_state_dict(checkpoint['model'])

    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased',
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    args.start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    args.end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    image = Image.open(os.path.join(args.data_dir, "images/CXR4_IM-2050-1001.png"))
    image = openi.val_transform(image).unsqueeze(0)
    caption, cap_mask = create_caption_and_mask(args.start_token, args.max_position_embeddings)

    output = evaluate(args, model, image, caption, cap_mask)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    print(result.capitalize())
