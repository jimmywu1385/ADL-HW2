import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from transformers import AutoTokenizer

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    split = args.split
    path = args.cache_dir / f"{split}.json"
    data = json.loads(path.read_text())

    context_max_len = args.max_len
    if split == "train" or split == "valid":
        output = []
        for i in data:
            center_ind = (context_max_len - 1) // 2
            tail_ind = center_ind - (context_max_len - 1) // 2
            head_ind = center_ind + context_max_len // 2 + 1
            answer_ind = i["answer"]["start"]
            answer = i["answer"]["text"]
            while not (answer_ind >= tail_ind and answer_ind+len(answer) <= head_ind):
                center_ind += (context_max_len // 2)
                tail_ind = center_ind - (context_max_len - 1) // 2
                head_ind = center_ind + context_max_len // 2 + 1
            if head_ind > len(i["paragraphs"][i["label"]]):
                head_ind = len(i["paragraphs"][i["label"]])

            paragraph = i["paragraphs"][i["label"]][tail_ind : head_ind]

            question_len = len(tokenizer.tokenize(i["question"]))
            start = question_len + 2 \
                    + len(tokenizer.tokenize(i["paragraphs"][i["label"]][tail_ind : answer_ind])) + 1 \
                    - 1
            end = start + (len(tokenizer.tokenize(answer)) - 1)
            output.append(
                {
                    "split" : split,
                    "id" : i["id"],   
                    "question" : i["question"],
                    "paragraph" : paragraph,
                    "start" : start,
                    "end" : end,
                }
            )
        with open(args.output_dir / f"{split}.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)      
    else:
        output = []
        for i in data:
            paragraph = []
            center_ind = (context_max_len - 1) // 2
            if center_ind > len(i["paragraphs"][i["label"]]):
                paragraph.append(i["paragraphs"][i["label"]])
            while center_ind < len(i["paragraphs"][i["label"]]):
                tail_ind = center_ind - (context_max_len - 1) // 2
                head_ind = center_ind + context_max_len // 2 + 1
                if head_ind <= len(i["paragraphs"][i["label"]]):
                    paragraph.append(i["paragraphs"][i["label"]][tail_ind : head_ind])
                else:
                    paragraph.append(i["paragraphs"][i["label"]][tail_ind : len(i["paragraphs"][i["label"]])])
                center_ind += (context_max_len // 2)
            output.append(
                {
                    "split" : split,
                    "id" : i["id"],   
                    "question" : i["question"],
                    "paragraph" : paragraph,
                }
            )
        with open(args.output_dir / f"{split}.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)      
    tokenizer.save_pretrained("./ckpt/tokenizer")  

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/context",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        help="Directory to save the tokenizer file.",
        default="hfl/chinese-macbert-large",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="file name",
        default="test",
    )
    parser.add_argument(
        "--max_len", type=int, help="sequence len", default=280
    ) 
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the output file.",
        default="./cache/QA/",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)