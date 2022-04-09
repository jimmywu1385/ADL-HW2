from typing import List, Dict

from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor

class contextData(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_len,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        if samples[0]["split"] == "train":
            labels = [i["label"] for i in samples]
            input_ids = []
            token_type_ids = []
            attention_mask = []
            for i in samples:
                input = []
                token_type = []
                attention = []
                question = i["question"]
                question_len = len(question)
                context_max = self.max_len - question_len - 3
                for j in i["paragraphs"]:
                    j = j[:context_max] if len(j) > context_max else j
                    tokens = self.tokenizer.encode_plus(question, j, 
                                            add_special_tokens=True, max_length=self.max_len,
                                            padding= 'max_length',
                                        )
                    input.append(tokens["input_ids"])
                    token_type.append(tokens["token_type_ids"])
                    attention.append(tokens["attention_mask"])
                
                input_ids.append(input)
                token_type_ids.append(token_type)
                attention_mask.append(attention)
            return {
                "input_ids" : LongTensor(input_ids),
                "token_type_ids" : LongTensor(token_type_ids),
                "attention_mask" : FloatTensor(attention_mask),
                "labels" : LongTensor(labels),
            }
        else:
            input_ids = []
            token_type_ids = []
            attention_mask = []
            for i in samples:
                input = []
                token_type = []
                attention = []
                question = i["question"]
                question_len = len(question)
                context_max = self.max_len - question_len - 3
                for j in i["paragraphs"]:
                    j = j[:context_max] if len(j) > context_max else j
                    tokens = self.tokenizer.encode_plus(question, j, 
                                            add_special_tokens=True, max_length=self.max_len,
                                            padding= 'max_length',
                                        )
                    input.append(tokens["input_ids"])
                    token_type.append(tokens["token_type_ids"])
                    attention.append(tokens["attention_mask"])
                
                input_ids.append(input)
                token_type_ids.append(token_type)
                attention_mask.append(attention)
            return {
                "input_ids" : LongTensor(input_ids),
                "token_type_ids" : LongTensor(token_type_ids),
                "attention_mask" : FloatTensor(attention_mask),
            }          