import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import string
from typing import Dict
from unittest import TestLoader

import torch
from tqdm import trange

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_datasets = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size, collate_fn=datasets["train"].collate_fn, shuffle=True)
    eval_datasets = torch.utils.data.DataLoader(datasets["eval"], batch_size=args.batch_size, collate_fn=datasets["eval"].collate_fn, shuffle=False)

    embeddings = torch.load(str(args.cache_dir) + "/embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
                    embeddings, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional, len(intent2idx), args.rnn_type
            ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0.0
        for i, dic in enumerate(train_datasets):
            text = dic["text"].to(args.device)
            intent = dic["intent"].to(args.device)

            optimizer.zero_grad()

            pred = model(text)
            loss = criterion(pred, intent)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 10 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f}")

        # TODO: Evaluation loop - calculate accuracy and save model weights
        print("------eval start------\n")

        model.eval()
        eval_loss = 0.0
        correct =0.0
        total =0.0
        for i, dic in enumerate(eval_datasets):
            text = dic["text"].to(args.device)
            intent = dic["intent"].to(args.device)

            with torch.no_grad():
                pred = model(text)
                loss = criterion(pred, intent)
            
            _, predicted = torch.max(pred, 1)
            total += intent.size(0)
            correct += (predicted == intent).sum().item()
            eval_loss += loss.item()
            if i % 10 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f}")

        print(f"Accuracy : {100*(correct/total)} %")
    
    print("DONE\n")

    # TODO: Inference on test set

    torch.save(model.state_dict(), args.ckpt_dir / (args.model_name+".pt"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="GRU")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    # save model
    parser.add_argument("--model_name", type=str, default="best")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)