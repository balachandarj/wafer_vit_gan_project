import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from data.wafer_dataset import WaferFolder

def get_loaders(root, img_size=128, batch_size=64, num_workers=2):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    ds_train = WaferFolder(root, split="train", img_size=img_size, transform=tf)
    ds_val = WaferFolder(root, split="val", img_size=img_size, transform=tf)
    ds_test = WaferFolder(root, split="test", img_size=img_size, transform=tf)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_val, dl_test

def save_ckpt(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
