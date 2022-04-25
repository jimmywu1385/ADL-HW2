import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

def main(args):
    with open(args.context_path, "r") as f:
        context = json.load(f)

    data = json.loads(args.data_path.read_text())

    output = []
    for i in data:
        paragraph = []
        for j in i["paragraphs"]:
            paragraph.append(context[j])
        if args.split == "test":
            output.append(
                {
                    "split" : "test",
                    "id" : i["id"],   
                    "question" : i["question"],
                    "paragraphs" : paragraph,
                }
            )
        else:
            ind = i["paragraphs"].index(i["relevant"])
            output.append(
                {
                    "split" : "train",
                    "id" : i["id"],   
                    "label" : ind,
                    "question" : i["question"],
                    "paragraphs" : paragraph,
                    "answer" : i["answer"],
                }
            )
    with open(args.output_dir / f"{args.split}.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)        

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the dataset.",
        default="./data/train.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the output file.",
        default="./cache/context/",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Directory to save the output file.",
        default="test",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)