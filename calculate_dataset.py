import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

from fid.inception import InceptionV3
from fid.fid_score import save_statistics, _compute_statistics_of_path

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument("save_path", type=str, help= ("Path to save .npz file of dataset"))

def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.path):
        raise RuntimeError('Invalid path: %s' % args.path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]

    model = InceptionV3([block_idx]).to(device)

    print(args.path, model, args.batch_size, args.dims, args.device)

    m, s = _compute_statistics_of_path(args.path, model, args.batch_size, args.dims, device)

    save_statistics(args.save_path, m, s)

if __name__ == '__main__':
    main()