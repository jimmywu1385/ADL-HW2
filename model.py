from typing import Any, Dict, List, Tuple
from pathlib import Path

import torch
from transformers import BertForMultipleChoice

class context_selector(torch.nn.Module):
    def __init__(
        self,
        pre_trained_path: Path,
        split: str,
    ) -> None:
        super(context_selector, self).__init__()
        self.pre_trained_path = pre_trained_path
        self.split = split
        if split == "train":
            self.bert = BertForMultipleChoice.from_pretrained(pre_trained_path)
        else:
            self.bert = BertForMultipleChoice

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None) -> Any:
        if self.split == "train":
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, labels=labels,
                        )
        else:
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask)            
        return output
