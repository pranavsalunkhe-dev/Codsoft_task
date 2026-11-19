# Image captioner

ResNet-50 (frozen, ImageNet-pretrained) extracts a 7x7 grid of visual features.
A 3-layer transformer decoder cross-attends to that grid and writes the caption
word by word. Served through a small Flask app with beam-search decoding.

## 1. Install
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## 2. Get data (Flickr8k)
Download the Flickr8k images and `Flickr8k.token.txt` (or the Kaggle `captions.txt`).
Put them in `data/`, e.g. `data/Flicker8k_Dataset/` and `data/Flickr8k.token.txt`.

## 3. Train
    python train.py --images data/Flicker8k_Dataset --captions data/Flickr8k.token.txt

A GPU is strongly recommended (~15 epochs). The best checkpoint by validation loss
is saved to `checkpoints/captioner.pt`.

## 4. Run the web app
    python app.py
Open http://127.0.0.1:5000

## Notes
- Flickr8k is small, so captions will be simple and sometimes generic. For better
  quality, train on Flickr30k or MS COCO (same file format), raise `--epochs`,
  or unfreeze the top ResNet block.
- Swap ResNet for VGG by changing `Encoder` in `model.py` (use the conv feature
  map, 512 channels, and update the projection size).
- Set `CKPT=path/to/file.pt` to load a different checkpoint, `PORT` to change the port.

## Files
- `vocab.py`    tokenizer and vocabulary
- `model.py`    encoder, decoder, beam search
- `train.py`    training and validation loop
- `app.py`      Flask API (`/api/caption`, `/api/health`)
- `templates/index.html`  web UI
