from transformers import BertTokenizer as t
to = t.from_pretrained("bert-base-chinese")
c = "幹你娘"
x = "雞排"
#c = to.encode_plus(c,x, add_special_tokens = True)
print(c)
to.save_pretrained("./q")