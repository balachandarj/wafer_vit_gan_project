import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.vit import ViTClassifier
from utils import get_loaders, save_ckpt

def evaluate(model, dl, device):
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            ys.append(y)
            yh.append(pred)
    ys = torch.cat([t for t in ys]).numpy()
    yh = torch.cat([t for t in yh]).numpy()
    return accuracy_score(ys, yh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", type=str, default="ckpt/vit.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, dl_val, dl_test = get_loaders(args.data, args.img_size, args.batch_size)

    model = ViTClassifier(img_size=args.img_size).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.item()))
        val_acc = evaluate(model, dl_val, device)
        print(f"Val Acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_ckpt({"model": model.state_dict(), "val_acc": best_val, "img_size": args.img_size}, args.save)

    test_acc = evaluate(model, dl_test, device)
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
