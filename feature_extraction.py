from PIL import Image
import torch
from conch.open_clip_custom import create_model_from_pretrained
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="Feature extraction w/ UNI")
parser.add_argument('--folder_path', type=str)
parser.add_argument('--folder_save', type=str)
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--random', action='store_true')
args = parser.parse_args()
folder_path = args.folder_path
folder_save = args.folder_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")
model.to(device)

# Data loading
list_wsi = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
list_wsi.reverse() if args.reverse else None
random.shuffle(list_wsi) if args.random else None

for name_wsi in list_wsi:
    file_npy = os.path.join(folder_save, name_wsi + ".npy")
    if os.path.isfile(file_npy):
        continue
    print(name_wsi)

    folder_wsi = os.path.join(folder_path, name_wsi)
    images_list = os.listdir(folder_wsi)
    images_list = [file for file in images_list if file.lower().endswith('.png')]
    images = [Image.open(os.path.join(folder_wsi, patch)) for patch in tqdm(images_list) if os.path.isfile(os.path.join(folder_wsi, patch))]

    patch_embeddings = []
    for img in tqdm(images):
        img =  preprocess(img).unsqueeze(dim=0).to(device)
        with torch.inference_mode():
            x = model.encode_image(img, proj_contrast=False, normalize=False).squeeze().cpu().numpy()
        patch_embeddings.append(x)
    patch_embeddings = np.stack(patch_embeddings)
    np.save(file_npy, patch_embeddings)