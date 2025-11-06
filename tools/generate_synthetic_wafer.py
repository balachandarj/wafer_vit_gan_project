import os, argparse
import numpy as np
from PIL import Image

def make_dir(d):
    os.makedirs(d, exist_ok=True)

def ring_pattern(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w//2, h//2
    rr = np.sqrt((xx-cx)**2 + (yy-cy)**2) / (min(h,w)/2)
    ring = np.exp(-((rr-0.6)**2)*25.0)
    return ring

def add_scratches(img, n=3):
    h, w = img.shape
    for _ in range(n):
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(h//4, h//2)
        angle = np.random.uniform(0, np.pi)
        xs = (x0 + np.arange(length)*np.cos(angle)).astype(int)
        ys = (y0 + np.arange(length)*np.sin(angle)).astype(int)
        xs = np.clip(xs, 0, w-1)
        ys = np.clip(ys, 0, h-1)
        img[ys, xs] += 0.8
    return img

def add_particles(img, n=15):
    h, w = img.shape
    for _ in range(n):
        cx, cy = np.random.randint(5, w-5), np.random.randint(5, h-5)
        r = np.random.randint(1, 4)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask = xx*xx + yy*yy <= r*r
        sub = img[cy-r:cy+r+1, cx-r:cx+r+1]
        if sub.shape == mask.shape:
            sub[mask] += 0.9
    return img

def save_img(path, arr):
    arr = np.clip(arr, 0, 1)
    img = (arr*255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

def gen_sample(label, img_size):
    base = np.zeros((img_size, img_size), dtype=np.float32)
    # wafer-like base
    base += ring_pattern(img_size, img_size) * 0.4
    if label == "normal":
        pass
    elif label == "scratch":
        base = add_scratches(base, n=np.random.randint(2, 5))
    elif label == "particle":
        base = add_particles(base, n=np.random.randint(8, 20))
    elif label == "edge_ring":
        base += ring_pattern(img_size, img_size) * 0.6
    # normalize
    base = (base - base.min())/(base.max() - base.min() + 1e-8)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--img_size", type=int, default=128)
    args = ap.parse_args()

    classes = ["normal", "scratch", "particle", "edge_ring"]
    for split in ["train", "val", "test"]:
        for c in classes:
            make_dir(os.path.join(args.out, split, c))

    rng = np.random.RandomState(123)
    ntrain, nval, ntest = args.n_per_class, max(args.n_per_class//5, 50), max(args.n_per_class//5, 50)

    for c in classes:
        # train
        for i in range(ntrain):
            arr = gen_sample(c, args.img_size)
            save_img(os.path.join(args.out, "train", c, f"{c}_{i:05d}.png"), arr)
        # val
        for i in range(nval):
            arr = gen_sample(c, args.img_size)
            save_img(os.path.join(args.out, "val", c, f"{c}_{i:05d}.png"), arr)
        # test
        for i in range(ntest):
            arr = gen_sample(c, args.img_size)
            save_img(os.path.join(args.out, "test", c, f"{c}_{i:05d}.png"), arr)

    print("Synthetic dataset generated at:", args.out)

if __name__ == "__main__":
    main()
