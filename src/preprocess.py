import sys
from os import listdir
from os.path import isfile, join
import random
import json
import logging

default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.StreamHandler())
default_logger.setLevel(logging.INFO)

def get_rawtext(txt_filepath):
    texts = []
    raw_text = ""
    with open(txt_filepath, encoding="utf-8", mode='r') as f:
        for line in f:
            if line[:5] == " "*5:
                text = raw_text.strip(" \t\n\r").replace('\n', ' ')
                if len(text)>80:
                    texts.append(text)
                raw_text = ""
            raw_text += line if line != "\n" else ""
    return texts

if __name__ == "__main__":
    source_dirpath = sys.argv[1] if len(sys.argv)>1 else "raw_data"
    default_logger.info(source_dirpath)

    filepaths = [join(source_dirpath, f) for f in listdir(source_dirpath) if isfile(join(source_dirpath, f))]
    default_logger.info(filepaths)
    
    texts = []
    for f in filepaths:
        texts.extend({"source":f, "text":x} for x in get_rawtext(f))    
    random.shuffle(texts)
    default_logger.info(f"Texts count is {len(texts)}")

    if len(sys.argv)>2:
        train_jsonfile = join(sys.argv[2], "train.jsonl")
        val_jsonfile = join(sys.argv[2], "val.json")
    else:
        train_jsonfile = "train.jsonl"
        val_jsonfile = "val.jsonl"
    
    train_part = 0.8 * len(texts)
    with open(train_jsonfile, encoding='utf-8', mode="w") as w_train, open(val_jsonfile, encoding='utf-8', mode="w") as w_val:
        for i,record in enumerate(texts):
            if i<train_part:
                w_train.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
            else:
                w_val.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

    