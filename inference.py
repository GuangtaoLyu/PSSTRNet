import os
import glob
import argparse
from tqdm import tqdm

import torch
from torchvision import transforms as T
from PIL import Image

from psstrnet import PSSTRNet


to_tensor = T.ToTensor()
to_pil_image = T.ToPILImage()


# Downsample the input image to reduce memory usage
def load_and_preprocess_image(image_path, size=(1024, 1024)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img = to_tensor(img).float()
    img = torch.unsqueeze(img, 0)
    return img


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process images for OCR using Doctr.')
    parser.add_argument('--image_source', '-i', type=str, required=True, help='Path to an image or an image folder.')
    parser.add_argument('--output_dir', '-o', type=str, default='./results', help='Directory to save the output images.')
    parser.add_argument('--model_path', '-m', type=str, default='./scut_syn.pth', help='Path to the model checkpoint.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (e.g., "cuda" or "cpu").')
    args = parser.parse_args()
    

    G = PSSTRNet()
    ckpt_dict = torch.load(args.model_path)
    G.load_state_dict(ckpt_dict['model_state_dict'])
    G = G.to(args.device)
    G.eval()

    image_source = args.image_source
    if os.path.exists(image_source):
        
        output_dir = args.output_dir
        mask_dir = os.path.join(output_dir, "masks")
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        if os.path.isdir(image_source):
            image_paths = glob.glob(os.path.join(image_source, "*"))
        else:
            image_paths = [image_source]
        
        for image_path in tqdm(image_paths):
            basename = os.path.basename(image_path).split(".")[0]
            image = load_and_preprocess_image(image_path)

            with torch.no_grad():
                image = image.to(args.device)
                _, _, _, str_out, _, _, _, mask_out = G(image)

            output_image = to_pil_image(str_out[0])
            mask = to_pil_image(mask_out[0])
            output_image.save(os.path.join(image_dir, f"{basename}.png"))
            mask.save(os.path.join(mask_dir, f"{basename}.png"))
    else:
        raise OSError(f"The provided input source dosen't exist.")