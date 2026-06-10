import os
import sys
import logging
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from tokenizer import FrenchTokenizer  

from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def build_english2french_dataset_in_memory(path_to_data_root, test_prop=0.005, cache_dir=None):
    logger.info("=== Phase 1: Loading Raw Text Files (Streaming Generator Mode) ===")
    
    all_dirs = [d for d in os.listdir(path_to_data_root) 
                if os.path.isdir(os.path.join(path_to_data_root, d)) and not d.startswith('.')]
    
    logger.info(f"Found {len(all_dirs)} data directories to scan.")
    
    french_files = []
    english_files = []
    
    for dir_name in all_dirs:
        path_to_dir = os.path.join(path_to_data_root, dir_name)
        for file in os.listdir(path_to_dir):
            if file.endswith(".fr"):
                french_files.append(os.path.join(path_to_dir, file))
            elif file.endswith(".en"):
                english_files.append(os.path.join(path_to_dir, file))
                
    if not french_files or not english_files:
        raise ValueError(f"Data mapping failed. Ensure paired .en and .fr text files exist within subdirectories.")

    logger.info(f"Targeting En Source: {english_files[0]}")
    logger.info(f"Targeting Fr Target: {french_files[0]}")

    def text_line_generator():
        with open(english_files[0], "r", encoding="utf-8") as f_en, \
             open(french_files[0], "r", encoding="utf-8") as f_fr:
            for line_en, line_fr in zip(f_en, f_fr):
                clean_en = line_en.strip()
                clean_fr = line_fr.strip()
                if clean_en and clean_fr:
                    yield {"english_src": clean_en, "french_tgt": clean_fr}

    logger.info("Building dataset layout dynamically via streaming generator...")
    
    combined_dataset = Dataset.from_generator(
        text_line_generator, 
        cache_dir=cache_dir,
        writer_batch_size=100_000
    )
    
    logger.info(f"Dataset successfully compiled! Total pairs: {len(combined_dataset)}")
    logger.info(f"Shuffling and dividing train/test arrays (Test split: {test_prop * 100}%)...")
    
    split_dataset = combined_dataset.train_test_split(
        test_size=test_prop, 
        writer_batch_size=500_000
    )
    
    return split_dataset
    
def tokenize_english2french_dataset_directly(raw_dataset, path_to_save, num_workers=24, truncate=False, max_length=512, min_length=5):
    logger.info("=== Phase 2: Tokenization Pipeline ===")
    
    logger.info("Initializing Tokenizer engines...")
    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json", truncate=truncate, max_length=max_length)
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def _tokenize_text(examples):
        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        
        src_ids = english_tokenizer(english_text, truncation=True, max_length=max_length)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
    
    logger.info(f"Spawning {num_workers} multi-threaded processes for global tokenization maps...")
    tokenized_dataset = raw_dataset.map(
        _tokenize_text, 
        batched=True, 
        num_proc=num_workers,
        desc="Tokenizing Src & Tgt languages"
    )
    
    logger.info("Stripping string schemas to save disk space...")
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])

    logger.info(f"Filtering out invalid sequences shorter than {min_length} tokens...")
    filter_func = lambda batch: [len(e) > min_length for e in batch["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(
        filter_func, 
        batched=True, 
        num_proc=num_workers,
        desc="Filtering short sequences"
    )
    
    logger.info(f"Writing final finalized tensor shards directly to: '{path_to_save}'...")
    tokenized_dataset.save_to_disk(path_to_save) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation Dataset Preparation")
    parser.add_argument("--test_split_pct", default=0.005, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--min_length", default=5, type=int)
    parser.add_argument("--path_to_data_root", required=True, type=str)
    parser.add_argument("--num_workers", default=24, type=int)
    parser.add_argument("--huggingface_cache_dir", default=None, type=str)

    args = parser.parse_args()
    
    path_to_data_tokenized = os.path.join(args.path_to_data_root, "tokenized_french2english_corpus")

    raw_memory_dataset = build_english2french_dataset_in_memory(
        path_to_data_root=args.path_to_data_root,
        test_prop=args.test_split_pct,
        cache_dir=args.huggingface_cache_dir
    )
    
    tokenize_english2french_dataset_directly(
        raw_dataset=raw_memory_dataset,
        path_to_save=path_to_data_tokenized,
        truncate=True,
        max_length=args.max_length,
        min_length=args.min_length,
        num_workers=args.num_workers
    )
    
