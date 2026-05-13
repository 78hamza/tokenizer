from bpe import BPEtokenizer
from spu import SPM


def main():
    try:
        filename: str = "sample.txt"
        bpe = BPEtokenizer(filename)
        texts = []
        with open(filename, "r+") as f:
            for line in f:
                texts.append(line)

        sp = SPM(texts)
        for text in texts:
            bpe.init_bpe_tokenizers()
            hf_tokens, hf_ids = bpe.hf_encode(text)
            sp.train_on_raw_text()
            sp.train_unigram_model()
            sp_tokens, sp_ids = sp.sp_uni_encode(text)
            print(f"Input: \"{text}\"")
            print("HF BPE Tokens: ", hf_tokens)
            print("HF BPE IDs: ", hf_ids)
            print("SP Unigram Tokens: ", sp_tokens)
            print("SP Unigram IDs: ", sp_ids)
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
