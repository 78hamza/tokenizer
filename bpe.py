from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPEtokenizer:
    def __init__(self, filename: str) -> None:
        self._filename: str = filename
        self.texts = []
        # Initialize a BPE Tokenizers (BPE)
        self.hf_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.hf_tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(vocab_size=50, special_tokens=["[UNK]"])
        self.hf_tokenizer.train_from_iterator(self.texts, self.trainer)

    # get the text
    def get_text(self):
        with open(self._filename, "r+") as f:
            for line in f:
                self.texts.append(line)

    # function to tokenize and detokenize
    def hf_encode(self, text):
        enc = self.hf_tokenizer.encode(text)
        return enc.tokens, enc.ids
