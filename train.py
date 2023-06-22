import argparse
import random
import json
import os
from typing import List, Dict, Tuple, Any
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import (logging,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          DataCollatorForTokenClassification,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer
)
from peft import prepare_model_for_int8_training
from dvclive.huggingface import DVCLiveCallback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


class SimpleTextGenerationDataset(Dataset):

    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        templates_path: str,
        max_size: int,
        overlap: int
    ):
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_size = max_size

        with open(templates_path) as r:
            self.templates = json.load(r)

        record_splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=overlap)
        self.inputs = []
        for record in tqdm(original_records):
            for chunk in record_splitter.split_text(record['text']):
                tensors = self.prepare_text(chunk)
                self.inputs.append(tensors)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]
    
    def prepare_text(self, text):
        templates = self.templates["prompts_input"]
        prompt_template = random.choice(templates)
        input = prompt_template.format(text=text)

        return self.prepare_tensors(input)
    
    def prepare_tensors(self, input):
        input_tokens = self.tokenizer(
            input,
            add_special_tokens=False,
            max_length = self.max_size,
            padding='max_length',
            truncation=False,
            return_tensors='pt'
        )
        if self.tokenizer.bos_token_id:
            input_tokens.insert(0, self.tokenizer.bos_token_id)
        input_ids = torch.LongTensor(input_tokens["input_ids"])
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_jsonl(file_name):
    with open(file_name, encoding='utf-8', mode='r') as r:
        return [json.loads(line) for line in r]

def train(
    config_file,
    train_file,
    val_file,
    output_dir,
    seed
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"
    trainer_config = config["trainer"]
    callbacks = [DVCLiveCallback]
    training_args = TrainingArguments(
        output_dir=output_dir,
        **trainer_config
    )

    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.save_pretrained(output_dir)

    train_records = read_jsonl(train_file)
    random.shuffle(train_records)
    val_records = read_jsonl(val_file)    

    templates_path = config.get("templates_path", "templates.json")
    train_dataset = SimpleTextGenerationDataset(
            train_records,
            tokenizer,
            templates_path=templates_path,
            max_size=config.get("max_size", 512),
            overlap=config.get("overlap", 0)
    )

    val_dataset = SimpleTextGenerationDataset(
            val_records,
            tokenizer,
            templates_path=templates_path,
            max_size=config.get("max_size", 512),
            overlap=config.get("overlap", 0)
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    #DataCollatorForTokenClassification(tokenizer)

    print(f"INPUT_IDS\n{data_collator([train_dataset[0], train_dataset[1]])['input_ids'][0]}")
    print(f"MASK\n{data_collator([train_dataset[0], train_dataset[1]])['attention_mask'][0]}")
    print(f"LABELS\n{data_collator([train_dataset[0], train_dataset[1]])['labels'][0]}")

    load_in_8bit = bool(config.get("load_in_8bit", False))
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            device_map=device_map
        )
    if load_in_8bit:        
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config.json")
    parser.add_argument("--train-file", type=str, default="train.jsonl")
    parser.add_argument("--val-file", type=str, default="val.jsonl")
    parser.add_argument("--output-dir", type=str, default="trained_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))