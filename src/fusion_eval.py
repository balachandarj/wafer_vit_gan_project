import argparse, os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from models.vit import ViTClassifier
from models.vae_gan import Encoder, Decoder, reparameterize, recon_loss
from utils import get_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth")
    ap.add_argument("--vit", type=str, required=True)
    ap.add_argument("--vaegan", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, dl_test = get_loaders(args.data, args.img_size, args.batch_size)

    # Load ViT
    ckpt_vit = torch.load(args.vit, map_location="cpu")
    model = ViTClassifier(img_size=ckpt_vit.get("img_size", args.img_size))
    model.load_state_dict(ckpt_vit["model"])
    model.to(device).eval()

    # Load VAE-GAN
    ckpt_vg = torch.load(args.vaegan, map_location="cpu")
    E = Encoder(); G = Decoder()
    E.load_state_dict(ckpt_vg["E"]); G.load_state_dict(ckpt_vg["G"])
    E.to(device).eval(); G.to(device).eval()

    ys, yhat, yhat_fused = [], [], []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()

            mu, logvar = E(x)
            z = reparameterize(mu, logvar)
            xr = G(z)
            rec = (x - xr).abs().mean(dim=[1,2,3]).cpu().numpy()

            # simple fusion: if reconstruction error is high, lower confidence
            conf = probs.max(dim=1).values.cpu().numpy()
            fused_conf = conf * np.exp(-rec / (rec.mean()+1e-8))
            fused_pred = preds.copy()
            # (optional) could threshold fused_conf to flag anomalies
            ys.append(y.numpy()); yhat.append(preds); yhat_fused.append(fused_pred)

    ys = np.concatenate(ys)
    yhat = np.concatenate(yhat)
    yhat_fused = np.concatenate(yhat_fused)

    print("=== ViT only ===")
    print(classification_report(ys, yhat, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ys, yhat))

    print("\n=== Fused (simple) ===")
    print(classification_report(ys, yhat_fused, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ys, yhat_fused))

    # Save a confusion matrix plot (optional)
    cm = confusion_matrix(ys, yhat_fused)
    fig = plt.figure(figsize=(5,4), dpi=150)
    plt.imshow(cm, cmap="gray")
    plt.title("Confusion Matrix (Fused)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/cm_fused.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
