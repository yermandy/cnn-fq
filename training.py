import torch
import numpy as np
import torch.optim as optim
import os
from time import time
import matplotlib.pyplot as plt
from dataset import ListDataset
from model.cnn_fq import model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import telegram


parser = argparse.ArgumentParser(description='CNN-FQ training')
parser.add_argument('--batch_size', default=90, type=int, help='Batch size')
parser.add_argument('--workers', default=16, type=int, help='Workers number')
parser.add_argument('--checkpoint', default='', type=str, help='Path to checkpoint file')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
args = parser.parse_args()


os.makedirs('resources', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)


def plot_error(val_errors, trn_errors, from_epoch=1, vline_each=999):
    epochs = np.arange(from_epoch, len(val_errors) + from_epoch)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for epoch in epochs:
        if epoch % vline_each == 0 and epoch != 0:
            plt.axvline(x=epoch, alpha=0.5, ls='--', c='black')
    ax.plot(epochs, val_errors, lw=2, label='Validation')
    ax.plot(epochs, trn_errors, lw=2, label='Training')
    fontsize = 14
    ax.legend(loc='upper right', fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('Error', fontsize=fontsize)
    img_path = 'results/plots/error.png'
    fig.savefig(img_path, dpi = 300, bbox_inches='tight')
    telegram.send_photo_telegram(img_path)
    plt.close()


def plot_criterial(Fs, Ls, from_epoch=1, vline_each=999):
    epochs = np.arange(from_epoch, len(Fs) + from_epoch)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for epoch in epochs:
        if epoch % vline_each == 0 and epoch != 0:
                plt.axvline(x=epoch, alpha=0.5, ls='--', c='black')
    ax.plot(epochs, Ls, lw=2, label='L(θ)')
    ax.plot(epochs, Fs, lw=2, label='F(θ,q)')
    fontsize = 14
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('Estimated likelihood', fontsize=fontsize)
    ax.legend(loc='lower right', fontsize=fontsize)
    img_path = 'results/plots/objective.png'
    fig.savefig(img_path, dpi = 300, bbox_inches='tight')
    telegram.send_photo_telegram(img_path)
    plt.close()


def init_q(labels, e=0.1):
    ## q => [ q(0,0,0) q(0,0,1) ... q(1,1,0) q(1,1,1) ]
    q = np.empty((labels.shape[0], 8))
    pos = [0 + e, 1 + e, 1 + e, 1 + e, 2 + e, 2 + e, 2 + e, 3 + e]
    neg = [3 + e, 2 + e, 2 + e, 2 + e, 1 + e, 1 + e, 1 + e, 0 + e]
    p = np.nonzero(labels == 1)[0]
    n = np.nonzero(labels == 0)[0]
    q[p] = pos
    q[n] = neg
    return q / np.sum(q, axis=1)[:, np.newaxis]


def create_dataset(triplets, faces, transform=None):
    triplets_len = triplets.shape[0]
    triplets_labels = triplets[:, 3]
    triplets_indecies = np.empty((triplets_len * 3), dtype=np.int)
    triplets_indecies[0::3] = triplets[:, 0]
    triplets_indecies[1::3] = triplets[:, 1]
    triplets_indecies[2::3] = triplets[:, 2]
    faces = faces[triplets_indecies]
    paths = faces[:, 0] # array containing paths to photos
    bbs = faces[:, 1:5].astype(np.int) # array with bounding boxes
    return ListDataset(paths, bbs, triplets_labels, transform=transform, path_to_images='images/casia')


def split_pX(pX):
    pA = pX[:, 0:2] ## p(a=0|A) p(a=1|A)
    pB = pX[:, 2:4] ## p(b=0|B) p(b=1|B)
    pC = pX[:, 4:6] ## p(c=0|C) p(c=1|C)
    return pA, pB, pC


def calculate_PyIabc(q, labels):
    ## p(y=1|a,b,c), (a,b,c) ∈ {0,1}^3
    p1Iabc = np.sum(q[labels == 1], axis=0) / np.sum(q, axis=0)
    p1Iabc = np.atleast_2d(p1Iabc)
    ## p(y=0|a,b,c), (a,b,c) ∈ {0,1}^3
    p0Iabc = np.sum(q[labels == 0], axis=0) / np.sum(q, axis=0)
    p0Iabc = np.atleast_2d(p0Iabc)
    ## p(y_i|a,b,c), ∀i ∈ |Y|, (a,b,c) ∈ {0,1}^3
    PyIabc = np.empty((labels.shape[0], 8))
    PyIabc[labels == 0] = p0Iabc
    PyIabc[labels == 1] = p1Iabc
    return PyIabc, p0Iabc, p1Iabc


def calculate_F(q, PyIabc, pX):
    F = np.empty((pX.shape[0], 8))
    pA, pB, pC = split_pX(pX)
    F[:, 0] = q[:, 0] * np.log((PyIabc[:, 0] * pA[:, 0] * pB[:, 0] * pC[:, 0]) / q[:, 0])
    F[:, 1] = q[:, 1] * np.log((PyIabc[:, 1] * pA[:, 0] * pB[:, 0] * pC[:, 1]) / q[:, 1])
    F[:, 2] = q[:, 2] * np.log((PyIabc[:, 2] * pA[:, 0] * pB[:, 1] * pC[:, 0]) / q[:, 2])
    F[:, 3] = q[:, 3] * np.log((PyIabc[:, 3] * pA[:, 0] * pB[:, 1] * pC[:, 1]) / q[:, 3])
    F[:, 4] = q[:, 4] * np.log((PyIabc[:, 4] * pA[:, 1] * pB[:, 0] * pC[:, 0]) / q[:, 4])
    F[:, 5] = q[:, 5] * np.log((PyIabc[:, 5] * pA[:, 1] * pB[:, 0] * pC[:, 1]) / q[:, 5])
    F[:, 6] = q[:, 6] * np.log((PyIabc[:, 6] * pA[:, 1] * pB[:, 1] * pC[:, 0]) / q[:, 6])
    F[:, 7] = q[:, 7] * np.log((PyIabc[:, 7] * pA[:, 1] * pB[:, 1] * pC[:, 1]) / q[:, 7])
    return F.sum()


def calculate_L(PyIabc, pX):
    q = calculate_q(PyIabc, pX)
    return np.sum(np.log(np.sum(q, axis=1)))


def calculate_q(PyIabc, pX):
    ## q => [ q(0,0,0) q(0,0,1) ... q(1,1,0) q(1,1,1) ]
    q = np.empty((pX.shape[0], 8))
    pA, pB, pC = split_pX(pX)
    q[:, 0] = PyIabc[:, 0] * pA[:, 0] * pB[:, 0] * pC[:, 0]
    q[:, 1] = PyIabc[:, 1] * pA[:, 0] * pB[:, 0] * pC[:, 1]
    q[:, 2] = PyIabc[:, 2] * pA[:, 0] * pB[:, 1] * pC[:, 0]
    q[:, 3] = PyIabc[:, 3] * pA[:, 0] * pB[:, 1] * pC[:, 1]
    q[:, 4] = PyIabc[:, 4] * pA[:, 1] * pB[:, 0] * pC[:, 0]
    q[:, 5] = PyIabc[:, 5] * pA[:, 1] * pB[:, 0] * pC[:, 1]
    q[:, 6] = PyIabc[:, 6] * pA[:, 1] * pB[:, 1] * pC[:, 0]
    q[:, 7] = PyIabc[:, 7] * pA[:, 1] * pB[:, 1] * pC[:, 1]
    return q


def calculate_error(P1Iabc, probs, labels):
    ## p(1|A,B,C) = sum{p(1|a,b,c) * p(a|A) * p(b|B) * p(c|C)}, where (a,b,c)∈(0,1)^3
    q = calculate_q(P1Iabc, probs)
    q = np.sum(q, axis=1)
    predictions = np.where(q > 0.5, 1, 0)
    error = np.mean(labels != predictions)
    return error


def calculate_alpha(q):
    ## alpha => [ ⍺(a=0) ⍺(a=1) β(b=0) β(b=1) c(c=0) γ(c=1) ]
    alpha = np.empty((q.shape[0], 6))
    alpha[:, 0] = np.sum(q[:, 0:4], axis=1)
    alpha[:, 1] = np.sum(q[:, 4:8], axis=1)
    alpha[:, 2] = np.sum(q[:, [0,1,4,5]], axis=1)
    alpha[:, 3] = np.sum(q[:, [2,3,6,7]], axis=1)
    alpha[:, 4] = np.sum(q[:, [0,2,4,6]], axis=1)
    alpha[:, 5] = np.sum(q[:, [1,3,5,7]], axis=1)
    # alpha = alpha.ravel()
    alpha = alpha.reshape(-1, 2)
    alpha = torch.tensor(alpha, dtype=torch.float32)
    return alpha


def forward_pass(net, loader, dataset_len, device):
    net.eval()

    ## p(a=0|A) p(a=1|A) p(b=0|B) p(b=1|B) p(c=0|C) p(c=1|C)
    probs = np.empty((dataset_len, 6))
    probs = probs.ravel()
    start = 0
    finish = 0

    run = time()
    batch_time = time()
    with torch.no_grad():
        for i, x in enumerate(loader):
            x = x.to(device)
            output = net(x).squeeze()
            out_len = output.shape[0]
            start = finish
            finish += out_len

            ## output -> p(x=1|X), where x∈{a,b,c} and X∈{A,B,C} resp.
            output = output.detach().cpu().numpy()
            
            probs[start*2+0:finish*2+0:2] = 1 - output
            probs[start*2+1:finish*2+1:2] = output
            if i % 100 == 0:
                print(f"-> processed {finish}/{dataset_len * 3} images in {time() - batch_time:.3f} sec")
                batch_time = time()
    probs[probs == 0] = 0.001
    probs[probs == 1] = 0.999
    probs = probs.reshape((-1, 6))
    print(f'Execution time: {time() - run:.3f} sec')
    return probs


def train(net : model):
    cnn_epochs = net.get_cnn_epochs(3)
    em_epochs = net.get_em_epochs(30)
    lr = net.get_lr(0.0001)
    params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True
    }

    device = next(net.parameters()).device
    trn    = np.genfromtxt('resources/casia_trn.csv', dtype=np.int, delimiter=',')
    val    = np.genfromtxt('resources/casia_val.csv', dtype=np.int, delimiter=',')
    faces  = np.genfromtxt('resources/casia_boxes_refined.csv', dtype=np.str, delimiter=',')

    # ! comment while training
    # trn = trn[:10]
    # val = val[:10]

    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    trn_dataset = create_dataset(trn, faces, transform=transform)
    trn_loader_1 = DataLoader(trn_dataset, **params, shuffle=True)
    trn_loader_2 = DataLoader(trn_dataset, **params)
    trn_n_img = len(trn_dataset)
    trn_len = int(len(trn_dataset) / 3)
    trn_labels = trn_dataset.get_labels()

    val_dataset = create_dataset(val, faces)
    val_loader = DataLoader(val_dataset, **params)
    val_len = int(len(val_dataset) / 3)
    val_labels = val_dataset.get_labels()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    net.set_optimizer_state_dict(optimizer)
    
    trn_errors = net.get_trn_errors()
    val_errors = net.get_val_errors()
    
    Fs = net.get_Fs()
    Ls = net.get_Ls()

    q = net.get_q(init_q, trn_labels)

    alpha = calculate_alpha(q)
    PyIabc, P0Iabc, P1Iabc = calculate_PyIabc(q, trn_labels)

    from_em_epoch = net.get_em_epoch()
    from_cnn_epoch = net.get_cnn_epoch()

    for em_epoch in range(from_em_epoch, em_epochs):

        ## M-step -> maximizing F(θ,q) w.r.t p_θ(x|X) and p_θ(y|a,b,c)

        for cnn_epoch in range(from_cnn_epoch, cnn_epochs):

            print(f"\nepoch #{em_epoch * cnn_epochs + cnn_epoch + 1}")
            net.train()
            
            start = 0
            finish = 0

            ## add soft weights
            trn_dataset.weights = alpha

            for batch_iter, (x, w) in enumerate(trn_loader_1):
                run = time()
                
                optimizer.zero_grad()

                x = x.to(device)
                output = net(x)

                out_len = output.shape[0]
                start = finish
                finish += out_len

                output = torch.squeeze(output)
                output = torch.clamp(output, 0.001, 0.999)
                
                ## move weights to device
                # w[:, 0] => [ ⍺(a=0) or β(b=0) or γ(c=0) ]
                # w[:, 1] => [ ⍺(a=1) or β(b=1) or γ(c=1) ]
                w = w.to(device)

                ## calculate loss
                # ⍺(a=0) * log(1 - p(a=1|A)) + ⍺(a=1) * log(p(a=1|A)) +
                # β(b=0) * log(1 - p(b=1|B)) + β(b=1) * log(p(b=1|B)) + 
                # γ(c=0) * log(1 - p(c=1|C)) + γ(c=1) * log(p(c=1|C))
                loss = (w[:, 0] * torch.log(1 - output) + w[:, 1] * torch.log(output)).sum()
                loss = -loss

                loss.backward()
                optimizer.step()

                print(f"-> processed {finish}/{trn_n_img} images in {time() - run:.3f} sec, loss {loss:.5f}")

            ## remove soft weights
            trn_dataset.weights = None
            
            print("Calculating training error ...")
            trn_probs = forward_pass(net, trn_loader_2, trn_len, device)
            trn_error = calculate_error(P1Iabc, trn_probs, trn_labels)
            trn_errors.append(trn_error)
            F = calculate_F(q, PyIabc, trn_probs)
            L = calculate_L(PyIabc, trn_probs)
            Fs.append(F)
            Ls.append(L)
            print(f"Training error {(trn_error * 100):.2f}% | F(Θ,q)={F:.2f} | L(Θ)={L:.2f}")


            print("Calculating validation error ...")
            val_probs = forward_pass(net, val_loader, val_len, device)
            val_error = calculate_error(P1Iabc, val_probs, val_labels)
            val_errors.append(val_error)
            print(f"Validation error {(val_error * 100):.2f}%")


            ## save parameters of the model
            print("Saving results ...")
            checkpoint_n = em_epoch * cnn_epochs + cnn_epoch + 1
            checkpoint = {
                'Fs': Fs, 'Ls': Ls, 'lr': lr, 'q': q,
                'trn_errors': trn_errors, 'val_errors': val_errors,
                'em_epoch': em_epoch, 'cnn_epoch': cnn_epoch + 1,
                'em_epochs': em_epochs, 'cnn_epochs': cnn_epochs,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'results/checkpoints/checkpoint_{checkpoint_n}.pt')
            plot_criterial(Fs, Ls, vline_each=cnn_epochs)
            plot_error(val_errors, trn_errors, vline_each=cnn_epochs)

        ## set next cnn epoch to 0
        from_cnn_epoch = 0

        ## E-step
        print("Calculating new q(a,b,c)")
        q = calculate_q(PyIabc, trn_probs)
        q = q / np.sum(q, axis=1)[:, np.newaxis]
        alpha = calculate_alpha(q)
        PyIabc, P0Iabc, P1Iabc = calculate_PyIabc(q, trn_labels)

        ## Decrease learning rate and restart Adam
        # lr /= 1.25
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        # optimizer = optim.SGD(net.parameters(), lr=lr)

        print("Saving results ...")
        checkpoint = {
            'Fs': Fs, 'Ls': Ls, 'lr': lr, 'q': q,
            'trn_errors': trn_errors, 'val_errors': val_errors,
            'em_epoch': em_epoch + 1, 'cnn_epoch': 0,
            'em_epochs': em_epochs, 'cnn_epochs': cnn_epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, f'results/checkpoints/checkpoint_{checkpoint_n}.pt')

        
if __name__ == '__main__':
    net = model(cuda=args.cuda, checkpoint_path=args.checkpoint)
    train(net)