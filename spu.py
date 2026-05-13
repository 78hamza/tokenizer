import sentencepiece as spm


class SPM:
    def __int__(self, texts: list) -> None:
        self.file = "sentences.txt"
        self.sp_uni = spm.SentencePieceTrainer(model_file='spm_unigram.model')
        self.texts: list = texts

    # Train on raw text file (required from the sentencepiece)
    def train_on_raw_text(self) -> bool:
        try:
            with open(self.file, "w+") as f:
                for line in self.texts:
                    f.write(line+"\n")
            return True
        except Exception:
            return False

    # train unigram model
    def train_unigram_model(self) -> None:
        spm.SentencePieceTrainer.train(
            input=self.file, model_prefix='spm_unigram',
            vocab_size=50, model_type='unigram'
        )

    def sp_uni_encode(self, text) -> tuple:
        return self.sp_uni.encode_as_pieces(text), self.sp_uni.encode_as_ids(text)
