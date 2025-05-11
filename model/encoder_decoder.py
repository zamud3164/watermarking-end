import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from model.feature_extractor import FeatureExtractor

class EncoderDecoder(nn.Module):
    """
    Implements De-END: Decoder -> Encoder -> Noiselayer -> Decoder.
    The first decoder extracts latent features from the cover image.
    The extracted latent features are concatenated with the watermark message
    before passing into the Encoder.
    The Encoder outputs a residual, which is added to the cover image to create the watermarked image.
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__()

        self.feature_extractor = FeatureExtractor(config)
        self.encoder = Encoder(config)
        self.noiser = noiser

        self.decoder = Decoder(config)  # Sharing weights with the first decoder

    def forward(self, image, message):
        """
        Forward pass through De-END:
        1. First decoder extracts latent features from the cover image.
        2. Concatenates latent features with the watermark message.
        3. Encodes the image using the concatenated input.
        4. Applies the noise layer.
        5. Final decoder extracts watermark from the noisy image.
        """
        latent_features = self.feature_extractor(image)  # Extract latent features from cover image

        latent_features = latent_features.view(latent_features.shape[0], latent_features.shape[1], 1, 1)  # [64, 30, 1, 1]
        message = message.view(message.shape[0], message.shape[1], 1, 1)  # [64, 30, 1, 1]

        # Concatenate latent features with the watermark message
        combined_input = torch.cat((latent_features, message), dim=1)

        #encoded_residual = self.encoder(image, combined_input, message)  # Encoder outputs residual
        encoded_residual = self.encoder(image, latent_features, message)
        encoded_image = image + encoded_residual  # Add residual to cover image

        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]

        decoded_message = self.decoder(noised_image)  # Final decoder extracts the watermark
        return encoded_image, noised_image, decoded_message
