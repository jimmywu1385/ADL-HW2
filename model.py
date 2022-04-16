from typing import Any, Dict, List, Tuple
from pathlib import Path
import pickle

import torch
from transformers import BertForMultipleChoice, AutoModelForQuestionAnswering, BertConfig

class context_selector(torch.nn.Module):
    def __init__(
        self,
        pre_trained_path: Path,
        split: str,
        config: Path = None,
    ) -> None:
        super(context_selector, self).__init__()
        self.pre_trained_path = pre_trained_path
        self.split = split
        if split == "test":
            with open(config, "rb") as f:
                self.config = pickle.load(f)
            self.bert = BertForMultipleChoice(self.config)
        else:
            self.bert = BertForMultipleChoice.from_pretrained(pre_trained_path)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None) -> Any:
        if self.split == "train":
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, labels=labels,
                        )
        else:
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask)            
        return output

class QA(torch.nn.Module):
    def __init__(
        self,
        pre_trained_path: Path,
        split: str,
        config: Path = None,
    ) -> None:
        super(QA, self).__init__()
        self.pre_trained_path = pre_trained_path
        self.split = split
        if split == "test":
            with open(config, "rb") as f:
                self.config = pickle.load(f)
            self.bert = AutoModelForQuestionAnswering.from_config(self.config)
        else:
            self.bert = AutoModelForQuestionAnswering.from_pretrained(pre_trained_path)

    def forward(self, input_ids, token_type_ids, attention_mask, start=None, end=None) -> Any:
        if self.split == "train":
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, start_positions=start,
                               end_positions=end, 
                        )
        else:
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask)            
        return output