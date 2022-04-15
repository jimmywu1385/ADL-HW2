import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import string
from typing import Dict
import tokenizers

import torch
from tqdm import trange
from transformers import BertTokenizer

from dataset import contextData
from model import context_selector

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)

    path = args.cache_dir / "test.json"
    data = json.loads(path.read_text())
    datasets = contextData(data, tokenizer, args.max_len)

    # TODO: crecate DataLoader for train / dev datasets
    test_datasets = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, collate_fn=datasets.collate_fn, shuffle=False)

    # TODO: init model and move model to target device(cpu / gpu)
    model = context_selector(args.pretrained_path, "test", args.ckpt_dir / Path("config.pkl")).to(args.device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    for i, dic in enumerate(test_datasets):
        input_ids = dic["input_ids"].to(args.device)
        token_type_ids = dic["token_type_ids"].to(args.device)
        attention_mask = dic["attention_mask"].to(args.device)

        with torch.no_grad():
            output = model(input_ids, token_type_ids, attention_mask)
            logits = output.logits
            data[i]["label"] = int(logits.argmax(1).item())
        print(i)
            
    path = args.cache_dir / "test.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/context",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/context",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        help="Directory to save the tokenizer file.",
        default="./ckpt/tokenizer",
    )
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="model.pt",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--max_len", type=int, help="sequence len", default=510
    )    
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )

    # model
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="model path.",
        default="bert-base-chinese",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)