import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing

special_token_dict = {
    "unknown_token": "[UNK]",
    "pad_token": "[PAD]",
    "start_token": "[BOS]",
    "end_token": "[EOS]"
}

def train_tokenizer(path_to_data_root):
    tokenizer = Tokenizer(WordPiece(unk_token= special_token_dict["unknown_token"]))
    tokenizer.normalizer = normalizers.Sequence([NFC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    french_files = glob.glob(os.path.join(path_to_data_root,"**/*.fr"))

    trainer = WordPieceTrainer(vocab_size= 32000, special_tokens= list(special_token_dict.values()))
    tokenizer.train(french_files, trainer)
    tokenizer.save("trained_tokenizer/french_wp.json")

class FrenchTokenizer:
    def __init__(self,path_to_vocab,truncate = False,max_length = 512):
        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.prepare_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.special_tokens_dict = {
            "UNK": self.tokenizer.token_to_id("[UNK]"),

        }

    def prepare_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decode = decoders.WordPiece()
        return tokenizer
    
    def encode(self,input):
        def _parse_process_tokenized(tokenized):
            if self.truncate:
                tokenized.truncate(self.max_len, direction = "right")
            tokenized = self.post_processor.process(tokenized)
            return tokenized.ids
    
        if isinstance(input,str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _parse_process_tokenized(tokenized)
        
        elif isinstance(input, (list,tuple)):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_parse_process_tokenized(t) for t in tokenized]
        return tokenized

    def decode(self,input,skip_special_tokens = True):
        if isinstance(input,list):
            if all(isinstance(item,list) for item in input):
                decoded = self.tokenizer.decode_batch(input,skip_special_tokens = skip_special_tokens)
            elif all(isinstance(item,int) for item in input):
                decoded = self.tokenizer.decode(input,skip_special_tokens = skip_special_tokens)
        return decoded
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Tokenizer Prep")

    parser.add_argument(
        "--path_to_data_root",
        required= True,
        help = "Path to store the final tokenized dataset",
        type = str
    )

    args = parser.parse_args()

    path_to_data_root = ""
    train_tokenizer(args.path_to_data_root)
    tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json")
    sentence = "Hello World!"
    enc = tokenizer.encode(sentence)
    print(enc)
    dec = tokenizer.decode(enc,skip_special_tokens= False)
    print(dec)