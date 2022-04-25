mkdir -p ckpt/context/
mkdir -p ckpt/QA/

wget https://www.dropbox.com/s/tpgcvg8h88o30t6/bert-base-chinese-mc.pt?dl=1 -O ckpt/context/bert-base-chinese-mc.pt
wget https://www.dropbox.com/s/0w4s0iqf6r9zykh/contextconfig.pkl?dl=1 -O ckpt/context/config.pkl
wget https://www.dropbox.com/s/rtbumqdezgjmg06/contextspecial_tokens_map.json?dl=1 -O cache/context/special_tokens_map.json
wget https://www.dropbox.com/s/9nq626bc662pp9m/contexttokenizer_config.json?dl=1 -O cache/context/tokenizer_config.json
wget https://www.dropbox.com/s/ovnpg6ys42fm86q/contextvocab.txt?dl=1 -O cache/context/vocab.txt

wget https://www.dropbox.com/s/890ubmqi1a0pdmk/macbert_QA.pt?dl=1 -O ckpt/QA/macbert_QA.pt
wget https://www.dropbox.com/s/vzszt4zbm11n3u5/QAconfig.pkl?dl=1 -O ckpt/QA/config.pkl
wget https://www.dropbox.com/s/kniz0o6tp2fkanh/QAspecial_tokens_map.json?dl=1 -O cache/QA/special_tokens_map.json
wget https://www.dropbox.com/s/zuf527nau5y70oi/QAtokenizer_config.json?dl=1 -O cache/QA/tokenizer_config.json
wget https://www.dropbox.com/s/opfqp5jbjpculay/QAtokenizer.json?dl=1 -O cache/QA/tokenizer.json
wget https://www.dropbox.com/s/6vopa8bbxkwcw7e/QAvocab.txt?dl=1 -O cache/QA/vocab.txt