import json
from argparse import ArgumentParser, Namespace
from ntpath import join
from pathlib import Path
from typing import Any, Dict
import csv

import torch
from tqdm import trange
from transformers import AutoTokenizer

from dataset import QAData
from model import QA

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    path = args.cache_dir / "test.json"
    data = json.loads(path.read_text())
    datasets = QAData(data[:20], tokenizer)

    # TODO: crecate DataLoader for train / dev datasets
    test_datasets = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, collate_fn=datasets.collate_fn, shuffle=False)

    # TODO: init model and move model to target device(cpu / gpu)
    model = QA(args.pretrained_path, "test", args.ckpt_dir / Path("config.pkl")).to(args.device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    id_list = []
    answer_list = []
    with torch.no_grad():
        for i, dic in enumerate(test_datasets):
            input_ids = dic["input_ids"].to(args.device)
            token_type_ids = dic["token_type_ids"].to(args.device)
            attention_mask = dic["attention_mask"].to(args.device)
            nums = dic["nums"]
            id = dic["id"]
            paragraphs = dic["paragraphs"]
            paragraph_offsets = dic["paragraph_offsets"]

            max_prob = [float("-inf")] * len(input_ids)
            answer = [""] * len(input_ids)
            for j in range(nums):
                output = model(input_ids[:,j,:], token_type_ids[:,j,:], attention_mask[:,j,:])
                start_prob, start_index = torch.max(output.start_logits, dim=-1)
                end_prob, end_index = torch.max(output.end_logits, dim=-1)

                for k in range(len(input_ids)):
                    prob = start_prob[k] + end_prob[k]

                    if prob > max_prob[k] and start_index[k] <= end_index[k] and end_index[k] - start_index[k] < 60:
                        max_prob[k] = prob
                        answer[k] = paragraphs[k][j][paragraph_offsets[k][j][start_index[k]][0] : paragraph_offsets[k][j][end_index[k]][1]]
                
            id_list += id
            answer_list += answer
    '''
    for i, dic in enumerate(test_datasets):
        input_ids = dic["input_ids"].to(args.device)
        token_type_ids = dic["token_type_ids"].to(args.device)
        attention_mask = dic["attention_mask"].to(args.device)
        nums = dic["nums"].to(args.device)
        id = dic["id"][0]
        paragraphs = dic["paragraphs"][0]
        paragraph_offsets = dic["paragraph_offsets"][0]

        max_prob = float("-inf")   
        with torch.no_grad():
            for j in range(nums.item()):
                output = model(input_ids[:,j,:], token_type_ids[:,j,:], attention_mask[:,j,:])
                start_prob, start_index = torch.max(output.start_logits, dim=-1)
                end_prob, end_index = torch.max(output.end_logits, dim=-1)

                prob = start_prob + end_prob

                if prob > max_prob and start_index <= end_index and end_index - start_index < 60:
                    max_prob = prob
                    answer = paragraphs[j][paragraph_offsets[j][start_index][0] : paragraph_offsets[j][end_index][1]]

        answer_list.append(answer.replace(" ", ""))
        id_list.append(id)
        print(i)
    '''        
    with open(args.pred_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['id', 'answer'])
        for i in range(len(id_list)):
            writer.writerow([id_list[i], answer_list[i]])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/QA",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/QA",
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
        default="hfl/chinese-macbert-large",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Directory to save the tokenizer file.",
        default="qq.csv",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)