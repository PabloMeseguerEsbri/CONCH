import argparse
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained

parser = argparse.ArgumentParser(description="Feature extraction w/ UNI")
parser.add_argument('--folder_path', type=str)
parser.add_argument('--folder_save', type=str)
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--data_loading', type=str, choices=["PATH", "CLAM", "AI4SKIN"])
parser.add_argument('--slicing', action='store_true')
parser.add_argument('--slicing_shape', type=int, choices=[512,256])
args = parser.parse_args()
folder_path = args.folder_path
folder_save = args.folder_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
model.to(device)

# Data loading
if args.data_loading == "CLAM":
    list_wsi = os.listdir(folder_path)
elif args.data_loading == "PATH":
    list_wsi = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
elif args.data_loading == "AI4SKIN":
    patches = pd.read_csv("./docs/csv/AI4SKIN_patches.csv")['images'].values
    list_wsi = pd.read_csv("./docs/csv/AI4SKIN_WSI.csv", delimiter=",")['WSI'].values
    patches_ids = np.array([patch[:9] for patch in patches])

list_wsi.reverse() if args.reverse else None
random.shuffle(list_wsi) if args.random else None

for name_wsi in list_wsi:
    try:
        if args.data_loading == "CLAM":
            name_wsi = name_wsi[:-3]
        file_npy = os.path.join(folder_save, name_wsi + ".npy")
        if os.path.isfile(file_npy):
            continue
        print(name_wsi)

        if args.data_loading == "CLAM":
            with h5py.File(os.path.join(folder_path, name_wsi + ".h5"), 'r') as file:
                images = file['imgs'][:]
                coords = file['coords'][:]
            if args.slicing:
                images, coords = slicing(size_out=args.slicing_shape, images=images, coords=coords)  # Slicing
            images = [Image.fromarray(patch) for patch in images]

        if args.data_loading == "PATH":
            folder_wsi = os.path.join(folder_path, name_wsi)
            images_list = os.listdir(folder_wsi)
            images_list = [file for file in images_list if file.lower().endswith('.png')]
            images = [Image.open(os.path.join(folder_wsi, patch)) for patch in tqdm(images_list) if os.path.isfile(os.path.join(folder_wsi, patch))]

        if args.data_loading == "AI4SKIN":
            img_files = patches[np.array(patches_ids) == name_wsi]  # All patches of one WSI
            if img_files.size == 0:
                print("WSI does not have any patch")
                continue
            if name_wsi.startswith("HCUV"):
                folder_wsi = os.path.join(folder_path , "Images/")
            elif name_wsi.startswith("HUSC"):
                folder_wsi = os.path.join(folder_path , "Images_Jose/")
            if img_files.size > 7500:
                img_files = img_files[:7500]
            images = [Image.open(folder_wsi + patch) for patch in tqdm(img_files) if os.path.isfile(os.path.join(folder_wsi, patch))]

        # Feature extraction
        patch_embeddings = []
        for img in tqdm(images):
            img =  preprocess(img).unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                x = model.encode_image(img, proj_contrast=False, normalize=False).squeeze().cpu().numpy()
            patch_embeddings.append(x)
        patch_embeddings = np.stack(patch_embeddings)
        np.save(file_npy, patch_embeddings)
    except:
        continue