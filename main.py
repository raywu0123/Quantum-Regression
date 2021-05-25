import os
os.environ['PYTHONHASHSEED'] = str(0)
import random
random.seed(0)

import numpy as np
np.random.seed(0)
import torch
torch.random.manual_seed(0)
from torch import nn

from parser import get_parser

from data import DistanceDataset
from model import MODEL_HUB


def run(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    training_data, testing_data, 
    batch_size: int,
):
    train_s1, train_s2, train_d = training_data
    p = np.random.permutation(len(train_d))
    train_s1, train_s2, train_d = train_s1[p], train_s2[p], train_d[p]
    loss_fn = nn.MSELoss()

    n_batch = len(train_d) // batch_size
    train_losses = []
    for i_batch in range(n_batch):
        batch_train_s1 = torch.from_numpy(train_s1[i_batch * batch_size: (i_batch + 1) * batch_size]).to(device=device, dtype=torch.cfloat)
        batch_train_s2 = torch.from_numpy(train_s2[i_batch * batch_size: (i_batch + 1) * batch_size]).to(device=device, dtype=torch.cfloat)
        batch_train_d = torch.from_numpy(train_d[i_batch * batch_size: (i_batch + 1) * batch_size]).float().to(device=device)
        batch_pred_d = model(batch_train_s1, batch_train_s2)
        
        loss = loss_fn(batch_pred_d, batch_train_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    test_s1, test_s2, test_d = testing_data
    n_batch = len(test_d) // batch_size
    test_losses = []
    with torch.no_grad():
        for i_batch in range(n_batch):
            batch_test_s1 = torch.from_numpy(test_s1[i_batch * batch_size: (i_batch + 1) * batch_size]).to(device=device, dtype=torch.cfloat)
            batch_test_s2 = torch.from_numpy(test_s2[i_batch * batch_size: (i_batch + 1) * batch_size]).to(device=device, dtype=torch.cfloat)
            batch_test_d = torch.from_numpy(test_d[i_batch * batch_size: (i_batch + 1) * batch_size]).float().to(device=device)
            batch_pred_d = model(batch_test_s1, batch_test_s2)
            
            loss = loss_fn(batch_pred_d, batch_test_d)
            test_losses.append(loss.item())
    
    print(f"train_loss: {np.mean(train_losses)}, test_loss: {np.mean(test_losses)}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    dataset = DistanceDataset(n_qubit=args.n_qubit, distance_measure=args.dist)

    print("Preparing data ...")
    training_data, testing_data = dataset.get_training_data(args.num_data), dataset.get_testing_data(args.num_data)
    
    device = "gpu:0" if torch.cuda.is_available() else "cpu"
    model = MODEL_HUB[args.model](args.n_qubit).to(device)
    optimizer = torch.optim.SGD(lr=args.lr, params=model.parameters())
    
    for i_epoch in range(args.n_epoch):
        print(f'epoch: {i_epoch}')
        run(model, optimizer, device, training_data, testing_data, args.batch_size)