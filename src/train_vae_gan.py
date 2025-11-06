import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.vae_gan import Encoder, Decoder, Discriminator, reparameterize, recon_loss, kl_div
from utils import get_loaders, save_ckpt

def gan_loss(pred, target_true):
    if target_true:
        return torch.nn.functional.softplus(-pred).mean()
    else:
        return torch.nn.functional.softplus(pred).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--save", type=str, default="ckpt/vaegan.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, dl_val, dl_test = get_loaders(args.data, args.img_size, args.batch_size)

    E = Encoder(args.z_dim).to(device)
    G = Decoder(args.z_dim).to(device)
    D = Discriminator().to(device)

    opt_E = optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device)

            # Encode/Decode
            mu, logvar = E(x)
            z = reparameterize(mu, logvar)
            x_rec = G(z)

            # Discriminator
            D_real = D(x)
            D_fake = D(x_rec.detach())
            loss_D = gan_loss(D_real, True) + gan_loss(D_fake, False)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator (Decoder) and VAE terms
            D_fake2 = D(x_rec)
            loss_recon = recon_loss(x_rec, x)
            loss_kl = kl_div(mu, logvar)
            loss_G = gan_loss(D_fake2, True) + loss_recon + 0.1*loss_kl

            opt_E.zero_grad(); opt_G.zero_grad()
            loss_G.backward()
            opt_E.step(); opt_G.step()

            pbar.set_postfix(loss_D=float(loss_D.item()), loss_G=float(loss_G.item()))

        # Simple val loss: reconstruction on val set
        with torch.no_grad():
            rec_val = 0.0; n=0
            for xv, _ in dl_val:
                xv = xv.to(device)
                mu, logvar = E(xv)
                z = reparameterize(mu, logvar)
                xr = G(z)
                rec_val += recon_loss(xr, xv).item()*xv.size(0)
                n += xv.size(0)
            rec_val /= max(n,1)
        print(f"Val Recon: {rec_val:.4f}")
        if rec_val < best_val:
            best_val = rec_val
            save_ckpt({"E": E.state_dict(), "G": G.state_dict(), "D": D.state_dict(), "val_recon": best_val, "img_size": args.img_size}, args.save)

    print("Training complete.")

if __name__ == "__main__":
    main()
