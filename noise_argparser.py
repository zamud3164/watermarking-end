import argparse
import re
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.median_blur import MedianBlur
from noise_layers.gaussian_noise import GaussianNoise



def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_gaussian(gaussian_command):
    # Matches gaussian(mean,stdmin,stdmax) or gaussian(stdmin,stdmax)
    match = re.match(r'gaussian\((\d+\.*\d*,\d+\.*\d*(?:,\d+\.*\d*)?)\)', gaussian_command)
    parts = match.groups()[0].split(',')

    if len(parts) == 2:
        std_min = float(parts[0])
        std_max = float(parts[1])
        return GaussianNoise(std_range=(std_min, std_max))
    elif len(parts) == 3:
        mean = float(parts[0])
        std_min = float(parts[1])
        std_max = float(parts[2])
        return GaussianNoise(mean=mean, std_range=(std_min, std_max))
    else:
        raise ValueError(f"Invalid gaussian command format: {gaussian_command}")


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))

def parse_medianblur(median_command):
    # Matches like: medianblur(3)
    match = re.match(r'medianblur\((\d+)\)', median_command)
    kernel_size = int(match.group(1))
    return MedianBlur(kernel_size)

def parse_jpeg(jpeg_command):
    match = re.match(r'jpeg\((\d+)\)', jpeg_command)
    if not match:
        raise ValueError(f"Invalid JPEG command format: {jpeg_command}")
    qf = int(match.group(1))
    return JpegCompression(quality_factor=qf)


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                layers.append(parse_cropout(command))
            elif command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command.startswith('medianblur'):
                layers.append(parse_medianblur(command))
            elif command[:len('gaussian')] == 'gaussian':
                layers.append(parse_gaussian(command))
            elif command[:len('jpeg')] == 'jpeg':
                layers.append(parse_jpeg(command))
            elif command[:len('quant')] == 'quant':
                layers.append('QuantizationPlaceholder')
            elif command[:len('identity')] == 'identity':
                # We are adding one Identity() layer in Noiser anyway
                pass
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)