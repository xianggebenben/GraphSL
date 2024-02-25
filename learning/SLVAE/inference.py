
# coding: utf-8

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam




def loss_seed(y, y_hat):
    loss = F.mse_loss(y_hat, y)
    return loss

def loss_seed_x(x, x_hat, loss_type='mse'):
    if loss_type == 'bce':
        return F.binary_cross_entropy(x_hat, x, reduction='mean')
    else:
        return F.mse_loss(x_hat, x)


def model_train(model, learning_rate, epochs, train_loader, batch_size, device):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, data_pair in enumerate(train_loader):
            input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)
            x = data_pair[:, :, 0].to(device)
            y = data_pair[:, :, 1].to(device)
            optimizer.zero_grad()

            x_hat, mean, log_var, y_hat = model(input_pair, x)
            loss = model.loss(x, x_hat, mean, log_var, y, y_hat)


            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
 
        print("\tEpoch", epoch+1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))

    print("Training Finished!!")
    model.eval()
    
    return model


def inference(deepis, train_set, test_set, learning_rate, epochs, device, loss_type='mse'):
    for p in deepis.parameters():
        p.requires_grad = False
        
    vae_model = deepis.vae_model
    gnn_model = deepis.gnn_model
    propagate = deepis.propagate
    
    x_comparison = {}
    
    for test_id, test in enumerate(test_set):
        rand_idx = np.random.randint(0, len(train_set) - 1)
        input_pair = torch.cat((train_set[rand_idx][:, 0], train_set[rand_idx][:, 1])).unsqueeze(0).to(device)
        x_true = test[:, 0].unsqueeze(0).to(device)
        y_true = test[:, 1].unsqueeze(0).to(device)
        x_hat, mean, log_var = vae_model(input_pair)
        # x_hat.requires_grad = True
        x_hat = Variable(x_hat.cuda(), requires_grad=True)
        # x_hat = x_hat.cuda().requires_grad_(True)
        
        input_optimizer = SGD([x_hat], lr=learning_rate)
        for epoch in range(epochs):
            input_optimizer.zero_grad()
            
            #x_hat.retain_grad()
            #x_hat.data.clamp_(0, 1)
            seed_idx = torch.LongTensor(np.argwhere(x_hat.cpu().detach().numpy() == 1)).to(device)
            
            y_hat = propagate(gnn_model(x_hat), seed_idx)
            
            loss = loss_seed(y_true, y_hat)
            
            loss.backward()
            input_optimizer.step()
            if (epoch+1)%5 == 0:
                print("Test #{} Epoch: {}".format(test_id+1, epoch+1),
                      "\tAverage y Loss: {:.6f}".format(loss.item()), 
                      "\tAverage x Loss: {:.6f}".format(loss_seed_x(x_true, x_hat, loss_type).item()))
                
        x_comparison[test_id] = [x_hat.cpu().detach().numpy(), x_true.cpu().detach().numpy()]
        print("Test #{} Completed!\n".format(test_id))
        
    return x_comparison
