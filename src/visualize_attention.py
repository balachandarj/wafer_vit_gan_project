import argparse, os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vit import ViTClassifier
from utils import get_loaders

def attention_rollout(model, x):
    # minimal proxy: use gradient * input on CLS token as saliency
    x.requires_grad_(True)
    logits = model(x)
    cls = logits.max(dim=1).values.sum()
    cls.backward()
    sal = x.grad.abs().mean(dim=1, keepdim=True)
    sal = (sal - sal.min())/(sal.max()-sal.min()+1e-8)
    return sal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth")
    ap.add_argument("--vit", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", type=str, default="outputs/attention")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, dl_test = get_loaders(args.data, args.img_size, args.batch_size)

    ckpt = torch.load(args.vit, map_location="cpu")
    model = ViTClassifier(img_size=ckpt.get("img_size", args.img_size))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        pass  # to freeze BN if any (none here)
    count = 0
    for x, y in dl_test:
        x = x.to(device)
        # enable grad for saliency
        x.requires_grad_(True)
        sal = attention_rollout(model, x)
        # overlay: stack [x, sal]
        for i in range(x.size(0)):
            inp = x[i]
            sa = sal[i]
            save_image(inp, os.path.join(args.out, f"img_{count:05d}.png"))
            save_image(sa, os.path.join(args.out, f"sal_{count:05d}.png"))
            count += 1
        if count >= 32:
            break
    print("Saved attention/saliency maps to", args.out)

if __name__ == "__main__":
    main()
