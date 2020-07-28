import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from time import time
from dataset import ListDataset
from model.cnn_fq import model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='CNN-FQ quality prediction')
parser.add_argument('--cuda', default=0, type=int, help='cuda device to run on')
parser.add_argument('--ref', default='resources/bounding_boxes.csv', type=str, help='path to CSV file with images and bouding boxes for quality prediction')
parser.add_argument('--images', default='images/casia', type=str, help='path to images folder')
parser.add_argument('--save_to', default='results/predictions', type=str, help='path to folder for output file with predictions')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--workers', default=16, type=int, help='numbers of workers')
parser.add_argument('--checkpoint', default='results/checkpoints/checkpoint.pt', type=str, help='Path to checkpoint file')
parser.add_argument('--uid', default='', type=str, help='Unique identifier for the output file')
parser.add_argument('--save_each', default=1, type=int, help='Output file saving frequency')
args = parser.parse_args()


os.makedirs(args.save_to, exist_ok=True)


def predict(net):
    params = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": True
    }

    device = next(net.parameters()).device
    faces = np.genfromtxt(args.ref, dtype=np.str, delimiter=',')
    images = faces[:, 0]
    boxes = faces[:, 1:5].astype(np.int)

    dataset = ListDataset(images, boxes, path_to_images=args.images)
    loader = DataLoader(dataset, **params)
    n_img = len(images)
    probs = np.empty(n_img, dtype=np.float32)
    start = 0
    finish = 0
    file_name = f'{args.save_to}/qualities{args.uid}.npy'

    net.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(loader, leave=False)):
            # run = time()
            x = x.to(device)
            output = net(x).squeeze()
            out_len = output.shape[0]
            start = finish
            finish += out_len
            output = output.detach().cpu().numpy()
            probs[start:finish] = output
            # print(f"-> processed {finish}/{n_img} images in {time() - run:.3f} seconds")
            if i % args.save_each == 0:
                np.save(file_name, probs)

    np.save(file_name, probs)


if __name__ == '__main__':
    net = model(cuda=args.cuda, checkpoint_path=args.checkpoint)
    predict(net)