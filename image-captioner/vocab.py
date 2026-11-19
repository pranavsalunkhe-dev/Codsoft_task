import re
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return _WORD.findall(text.lower())


class Vocabulary:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, captions, min_freq=5):
        counts = Counter(w for c in captions for w in tokenize(c))
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        return cls([PAD, START, END, UNK] + words)

    @classmethod
    def from_itos(cls, itos):
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=30):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(text)][:max_len]
        return [self.stoi[START]] + ids + [self.stoi[END]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == END:
                break
            if w in (PAD, START, UNK):
                continue
            words.append(w)
        return " ".join(words)
