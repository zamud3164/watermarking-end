import os
import argparse
import torch
import logging
import sys
import utils
import numpy as np
from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from utils_adaptor_loss import compute_psnr, compute_ssim, compute_ber
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import glob
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def load_images_from_folder(folder, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images) if images else None

def find_checkpoint_file(checkpoints_dir, epoch):
    pattern = os.path.join(checkpoints_dir, f'*epoch-{epoch}.pyt')
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {checkpoints_dir}")
    return matching_files[0]

def evaluate(model, device, test_images, batch_size, message_length, save_image_dir, save_count):
    model.encoder_decoder.eval()
    model.discriminator.eval()
    total_losses = {
        'encoder_mse': 0,
        'decoder_mse': 0,
        'bitwise_error': 0,
        'ssim': 0,
        'ber': 0,
        'adv_loss': 0
    }

    num_batches = 0
    first_batch = True
    with torch.no_grad():
        for i in range(0, test_images.size(0), batch_size):
            images = test_images[i:i+batch_size].to(device)
            messages = torch.randint(0, 2, (images.size(0), message_length), dtype=torch.float32, device=device)
            encoded_images, noised_images, decoded_messages = model.encoder_decoder(images, messages)

            if first_batch and save_image_dir is not None:
                os.makedirs(save_image_dir, exist_ok=True)
                num_to_save = min(save_count, images.size(0))
                for idx in range(num_to_save):
                    vutils.save_image(images[idx], os.path.join(save_image_dir, f"orig_{idx}.png"))
                    vutils.save_image(encoded_images[idx], os.path.join(save_image_dir, f"encoded_{idx}.png"))
                first_batch = False

            encoder_mse = torch.nn.functional.mse_loss(encoded_images, images).item()
            decoder_mse = torch.nn.functional.mse_loss(decoded_messages, messages).item()
            psnr_value = compute_psnr(images, encoded_images, max_psnr=50).item()
            ssim_value = compute_ssim(images, encoded_images).item()
            ber_value = compute_ber(messages, decoded_messages).item()

            decoded_rounded = decoded_messages.cpu().numpy().round().clip(0, 1)
            bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.cpu().numpy())) / messages.numel()

            d_target_label_encoded = torch.full((images.size(0), 1), model.encoded_label, device=device)
            d_on_encoded = model.discriminator(encoded_images)
            adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded.float()).item()

            total_losses['encoder_mse'] += encoder_mse
            total_losses['decoder_mse'] += decoder_mse
            total_losses['ssim'] += ssim_value
            total_losses['ber'] += ber_value
            total_losses['bitwise_error'] += bitwise_avg_err
            total_losses['adv_loss'] += adv_loss

            num_batches += 1

    return {key: val / num_batches for key, val in total_losses.items()}

def main():
    parser = argparse.ArgumentParser(description='Test HiDDeN Model')
    parser.add_argument('--checkpoint-folder', '-f', required=True, type=str, help='Path to the trained model checkpoint folder')
    parser.add_argument('--test-data', '-d', required=True, type=str, help='Path to the test dataset')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='Batch size for testing')
    parser.add_argument('--model-epoch', type=int, required=True, help='Epoch number to load full model checkpoint')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = find_checkpoint_file(os.path.join(args.checkpoint_folder, 'checkpoints'), args.model_epoch)
    model_ckpt = torch.load(checkpoint_path, map_location=device)

    options_file = os.path.join(args.checkpoint_folder, 'options-and-config.pickle')
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    hidden_config.use_adaptor = True

    model = Hidden(hidden_config, device, Noiser(noise_config, device), None)

    # Load all model components from checkpoint
    enc_dec_weights = model_ckpt['enc-dec-model']
    model.encoder_decoder.load_state_dict(enc_dec_weights)
    model.discriminator.load_state_dict(model_ckpt['discrim-model'])

    print(f"Loaded full model from checkpoint: {checkpoint_path}")

    test_images = load_images_from_folder(args.test_data, hidden_config.H)
    if test_images is None:
        print("No images found in the test dataset folder.")
        return

    print(f"Loaded {test_images.size(0)} test images.")

    save_image_dir = os.path.join(args.checkpoint_folder, 'test_images')
    save_count = 15

    test_metrics = evaluate(model, device, test_images, args.batch_size, hidden_config.message_length, save_image_dir, save_count)

    print("\n=== Test Results ===")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

if __name__ == '__main__':
    main()
