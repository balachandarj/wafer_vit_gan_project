import os, glob
from PIL import Image
from torch.utils.data import Dataset
import torch

CLASS_NAMES = ["normal", "scratch", "particle", "edge_ring"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

class WaferFolder(Dataset):
    def __init__(self, root, split="train", img_size=128, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        for c in CLASS_NAMES:
            files = sorted(glob.glob(os.path.join(root, split, c, "*.png")))
            self.samples += [(f, CLASS_TO_IDX[c]) for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            x = self.transform(img)
        else:
            # default: to tensor in [0,1]
            x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()/255.0)).view(img.size[1], img.size[0], 1).permute(2,0,1)
        return x, y
