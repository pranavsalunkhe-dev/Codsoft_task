import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from vocab import START, END, UNK

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([
    T.Resize(256), T.RandomCrop(224),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])


class Encoder(nn.Module):
    """Frozen ResNet-50 -> 7x7 grid of 2048-d features -> projected to d_model."""

    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        self.cnn = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(2048, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.zeros(1, 49, d_model))  # learned spatial position

    def train(self, mode=True):
        super().train(mode)
        self.cnn.eval()  # keep BatchNorm statistics frozen
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.cnn(x)  # B, 2048, 7, 7
        f = f.flatten(2).transpose(1, 2)  # B, 49, 2048
        return self.norm(self.proj(f) + self.pos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Decoder(nn.Module):
    """Transformer decoder: causal self-attention over words, cross-attention over image regions."""

    def __init__(self, vocab_size, d_model=512, heads=8, layers=3, ff=2048, dropout=0.1, max_len=40):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_len + 2)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, memory):
        T_ = tokens.size(1)
        causal = torch.triu(torch.full((T_, T_), float("-inf"), device=tokens.device), diagonal=1)
        x = self.pe(self.embed(tokens) * math.sqrt(self.d_model))
        return self.out(self.norm(self.tf(x, memory, tgt_mask=causal)))


class Captioner(nn.Module):
    def __init__(self, vocab_size, d_model=512, heads=8, layers=3, ff=2048, max_len=40, pretrained=True):
        super().__init__()
        self.encoder = Encoder(d_model, pretrained)
        self.decoder = Decoder(vocab_size, d_model, heads, layers, ff, max_len=max_len)

    def forward(self, images, tokens):
        return self.decoder(tokens, self.encoder(images))

    @torch.no_grad()
    def generate(self, image, vocab, beam=3, max_len=30, n_best=3):
        """Beam search. image: (1,3,224,224). Returns [(caption, avg_logprob), ...] best first."""
        dev = image.device
        mem = self.encoder(image)
        start, end, unk = vocab.stoi[START], vocab.stoi[END], vocab.stoi[UNK]
        beams, done = [(0.0, [start])], []
        for _ in range(max_len):
            cand = []
            for score, seq in beams:
                logits = self.decoder(torch.tensor([seq], device=dev), mem)[0, -1]
                logits[unk] = float("-inf")  # never emit <unk>
                logp = F.log_softmax(logits, -1)
                top = logp.topk(beam)
                for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((score + lp, seq + [idx]))
            cand.sort(key=lambda c: c[0], reverse=True)
            beams = []
            for s, q in cand[:beam]:
                (done if q[-1] == end else beams).append((s, q))
            if not beams:
                break
        ranked = sorted(done + beams, key=lambda c: c[0] / len(c[1]), reverse=True)
        return [(vocab.decode(q), s / len(q)) for s, q in ranked[:n_best]]
