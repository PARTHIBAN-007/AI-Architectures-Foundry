import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
import argparse

from tokenizers import FrenchTokenizer
from transformers import AutoTokenizer

def build_english2French_dataset(path_to_data_root,path_to_save,test_prop=0.005,cache_dir = None):
    hf_dataset = []
    for dir in os.listdir(path_to_data_root):
        path_to_dir = os.path.join(path_to_data_root,dir)
        if os.path.isdir(path_to_dir):
            print(f"Processing: {path_to_dir}")
            french_text = english_text = None
            for txt in os.listdir(path_to_dir):
                if txt.endswith(".fr"):
                    french_text = os.path.join(path_to_dir,txt)
                elif txt.endswith(".en"):
                    english_text = os.path.join(path_to_dir,txt)

            if french_text is not None and english_text is not None:
                french_dataset  = load_dataset("text", data_files = french_text, cache_dir = cache_dir)["train"]
                english_dataset = load_dataset("text",data_files = english_text, cache_dir = cache_dir)["train"]
            
                english_dataset = english_dataset.rename_column("text","english_src")
                english_dataset = english_dataset.add_column("french_tgt",french_dataset["text"])

                hf_dataset.append(english_dataset)
            
        hf_dataset - concatenate_datasets(hf_dataset)
        hf_dataset = hf_dataset.train_test_split(test_size = test_prop)
        hf_dataset.save_to_disk(path_to_save)
    
def tokenize_englsih2french_dataset(path_to_hf_data,path_to_save,num_workers = 24, truncate = False,max_length = 512, min_length = 5):
    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json",truncate = truncate,max_length = max_length)
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    raw_dataset = load_from_disk(path_to_hf_data)

    def _tokenize_text(examples):
        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        src_ids = english_tokenizer(english_text,truncate = True, max_length = 512)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
    
    tokenized_dataset = raw_dataset.map(_tokenize_text,batched = True, num_proc = num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src","french_tgt"])

    filter_func = lambda batch: [len(e) > min_length for e in batch["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(filter_func,batched = True)
    tokenized_dataset.save_to_disk(path_to_save) 


if __name__== "__main__":
    parser = argparse.ArgumentParser(description= " Translation Dataset Preparation")
    parser.add_argument(
        "--test_split_pct",
        default= 0.005,
        help = "What percent of data do you want to use for Train/Test split",
        type = float
    )

    parser.add_argument(
        "--max_length",
        default= 512,
        help = "pass in argument to override the default in Config",
        type = int
    )

    parser.add_argument(
        "--min_length",
        default= 5,
        help = "Removes any token sequences that are shorter than this length",
        type = int
    )

    parser.add_argument(
        "--path_to_data_root",
        required= True,
        help = "Path to store the tokenized dataset",
        type = str
    )

    parser.add_argument(
        "--num_workers",
        default= 24,
        help = "Number of Workers to use process dataset",
        type = int
    )

    args = parser.parse_args()
    path_to_data_root = args.path_to_data_root
    path_to_data_raw = os.path.join(path_to_data_root,"raw_english2french_corpus")
    path_to_data_tokenized = os.path.join(path_to_data_root, "tokenized_englsih2french_corpus")
    cache_dir = args.huggingface_cache_dir

    build_english2French_dataset(path_to_data_root,
                                 path_to_data_raw,
                                 test_prop= args.test_split_pct,
                                 cache_dir=cache_dir)
    
    tokenize_englsih2french_dataset(path_to_data_raw,
                                    path_to_data_tokenized,
                                    truncate = True,
                                    max_length = args.max_length,
                                    min_length= args.min_length,
                                    num_workers = args.num_workers)