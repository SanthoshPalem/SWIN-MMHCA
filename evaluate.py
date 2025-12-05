
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lpips import LPIPS
import numpy as np
from PIL import Image

# It seems there is an issue with the project structure, so I'm adding the project root to the python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data.dataloader import IXIDataset
from src.models.swin_mmhca import SwinMMHCA

def evaluate(model_path, data_path, device):
    # --- Model ---
    model = SwinMMHCA(
        img_size=64,
        patch_size=4,
        in_chans=1,
        embed_dim=192,
        depths=[2, 2],
        num_heads=[6, 6],
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        ape=True,
        patch_norm=True
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Dataset ---
    val_dataset = IXIDataset(data_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Metrics ---
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips = LPIPS(net='alex').to(device)

    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(val_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            sr_img = model(lr_img)
            
            # Clamp images to [0, 1] range for metrics
            sr_img = torch.clamp(sr_img, 0, 1)
            hr_img = torch.clamp(hr_img, 0, 1)

            psnr_total += psnr(sr_img, hr_img)
            ssim_total += ssim(sr_img, hr_img)
            lpips_total += lpips(sr_img, hr_img)

    avg_psnr = psnr_total / len(val_loader)
    avg_ssim = ssim_total / len(val_loader)
    avg_lpips = lpips_total / len(val_loader)

    print(f'PSNR: {avg_psnr.item():.4f}')
    print(f'SSIM: {avg_ssim.item():.4f}')
    print(f'LPIPS: {avg_lpips.item():.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(args.model_path, args.data_path, device)
