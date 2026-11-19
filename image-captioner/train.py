"""Train the captioner on Flickr8k (or any dataset in the same caption-file format).

  python train.py --images data/Flicker8k_Dataset --captions data/Flickr8k.token.txt

Caption file formats accepted:
  Flickr8k.token.txt : "img.jpg#0<TAB>A dog runs on grass ."
  captions.txt (CSV) : "img.jpg,A dog runs on grass ."   (header line "image,caption" is skipped)
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Captioner, EVAL_TF, TRAIN_TF
from vocab import Vocabulary


def read_captions(path):
    caps = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            key, text = line.split("\t", 1)
        elif "," in line:
            key, text = line.split(",", 1)
        else:
            continue
        key = key.split("#")[0].strip()
        if key.lower() in ("image", "image_name"):
            continue
        caps[key].append(text.strip().strip('"'))
    return caps


class CaptionData(Dataset):
    def __init__(self, root, pairs, vocab, transform, max_len):
        self.root, self.pairs, self.vocab, self.tf, self.max_len = Path(root), pairs, vocab, transform, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        name, text = self.pairs[i]
        img = Image.open(self.root / name).convert("RGB")
        return self.tf(img), torch.tensor(self.vocab.encode(text, self.max_len))


def collate(batch):
    imgs, caps = zip(*batch)
    padded = torch.zeros(len(caps), max(len(c) for c in caps), dtype=torch.long)
    for i, c in enumerate(caps):
        padded[i, : len(c)] = c
    return torch.stack(imgs), padded


def run_epoch(model, loader, crit, device, opt=None, sched=None, scaler=None):
    training = opt is not None
    model.train(training)
    total, count = 0.0, 0
    for imgs, caps in tqdm(loader, leave=False):
        imgs, caps = imgs.to(device), caps.to(device)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(imgs, caps[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
        if training:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        total += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", default="checkpoints/captioner.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    caps = read_captions(args.captions)
    names = sorted(n for n in caps if (Path(args.images) / n).exists())
    random.shuffle(names)
    cut = int(len(names) * 0.95)  # split by image so val images are never seen in training
    train_names, val_names = names[:cut], names[cut:]
    train_pairs = [(n, c) for n in train_names for c in caps[n]]
    val_pairs = [(n, c) for n in val_names for c in caps[n]]
    print(f"{len(train_names)} train images / {len(val_names)} val images")

    vocab = Vocabulary.build([c for _, c in train_pairs], args.min_freq)
    print(f"vocabulary: {len(vocab)} words")

    mk = lambda pairs, tf, shuffle: DataLoader(
        CaptionData(args.images, pairs, vocab, tf, args.max_len), batch_size=args.batch,
        shuffle=shuffle, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda")
    train_dl, val_dl = mk(train_pairs, TRAIN_TF, True), mk(val_pairs, EVAL_TF, False)

    config = dict(d_model=512, heads=8, layers=3, ff=2048, max_len=args.max_len + 5)
    model = Captioner(len(vocab), **config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, crit, device, opt, sched, scaler)
        va = run_epoch(model, val_dl, crit, device)
        print(f"epoch {epoch:2d}  train {tr:.3f}  val {va:.3f}")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "itos": vocab.itos, "config": config}, args.out)
            print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
