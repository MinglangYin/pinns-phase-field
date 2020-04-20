"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - h**2*u_xx + u^3 - u = 0, x in [-1, 1], t in [0, 1]
    with
        u(0, x) = tanh(x/(sqrt(2)*h))
        u_x(t, -1) = u_x(t, 1) = 0

    Input: [t, x]
    Output: [u]
"""

import sys
sys.path.insert(0,'../../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import gc
# from plotting import newfig, savefigf

torch.manual_seed(1234)
np.random.seed(1234)

eps = 0.03
theta = 0.1
epsilon = 1e-8

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 10, bias=False))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,3):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(10, 10, bias=False))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
        self.net.add_module('linear_layer_50', nn.Linear(10, 1, bias=False))

        # self.weight_bc = torch.tensor([1], requires_grad=False, dtype=torch.float32).to(device)
        # self.weight_ic = torch.tensor([100], requires_grad=False, dtype=torch.float32).to(device)

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:, 0], u_g[:, 1]
        u_gg = gradients(u_x, x)[0]
        u_xx = u_gg[:,1]
        loss = u_t - eps**2*u_xx + (u**3 - u)
        return (loss**2).mean()

    def loss_bc(self, x):
        u = self.net(x)
        u_x = gradients(u, x)[0][:, 1]
        return (u_x**2).mean()

    def loss_ic(self, x, u_init):
        u = self.net(x)
        return ((u - u_init)**2).mean()

    def grad_weights(loss):
        grad_w_list = []
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                grad = gradients(loss, param)[0]
                grad_w_list.append((grad.view(-1)))

        grad_w_tensor = grad_w_list[0]
        for i in range(1, len(grad_w_list)):
            torch.cat((grad_w_tensor, grad_w_list[i]))
            # print(len(grad_w_tensor))
        return torch.abs(grad_w_tensor.mean())

# def update_weights(loss_pde, loss_bc, loss_ic):
#     loss_ic_grad = grad_weights(loss_ic)
#     loss_bc_grad = grad_weights(loss_bc)
#     loss_pde_grad = grad_weights(loss_pde)

#     weight_bc = 0.9*weight_bc + 0.1*loss_pde_grad/(loss_ic_grad + epsilon)
#     weight_ic = 0.9*weight_ic + 0.1*loss_pde_grad/(loss_bc_grad + epsilon)

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def init_cond(x):
    return np.tanh(x[:,1]/(np.sqrt(2)*eps) )

def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    epochs = 10000
    num_x = 100
    num_t = 10
    lr = 0.001
    
    ##
    x = np.linspace(-1, 1, num=num_x)
    t = np.linspace(0, 1, num=num_t)
    x_grid, t_grid = np.meshgrid(x, t)
    x_f = np.concatenate((t_grid[1:-1,1:-1].flatten()[:,None], x_grid[1:-1,1:-1].flatten()[:,None]), axis=1)
    x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None]), axis=1)

    ##
    x_init = np.concatenate((np.zeros((num_x, 1)), x[:, None]), axis=1 )
    x_b_l = np.concatenate((t_grid[:,0][:,None], x_grid[:,0][:,None]), axis=1)
    x_b_r = np.concatenate((t_grid[:,-1][:,None], x_grid[:,-1][:,None]), axis=1)
    x_b = np.concatenate((x_b_l, x_b_r))

    ## Initial Condition
    u_init = init_cond(x_init)[:,None]

    # ## Form data tensor and send
    x_i_train = torch.tensor(x_init,requires_grad=True, dtype=torch.float32).to(device)
    x_b_train = torch.tensor(x_b, requires_grad=True, dtype=torch.float32).to(device)
    x_f_train = torch.tensor(x_f, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)

    ## instantiate model
    model = Model(device).to(device)

    weight_list = []
    bias_list = []
    for name, param in model.net.named_parameters():
        if param.requires_grad:
            if 'weight' in name:
                weight_list.append(param)
            if 'bias' in name:
                bias_list.append(param)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    print('start training...')
    tic = time.time()
    
    a_old = 1
    b_old = 100
    for epoch in range(1, epochs + 1):    
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_f_train)
            loss_bc = model.loss_bc(x_b_train)
            loss_ic = model.loss_ic(x_i_train, u_i_train)

            loss_pde.backward(retain_graph=True)
            w_pde_grad = [weight_list[i].grad for i in range(len(weight_list))]
            w_pde_grad_norm = np.mean([abs(to_numpy(w_pde_grad[i].detach()).flatten()).mean() for i in range(len(w_pde_grad))])
            print(w_pde_grad_norm)
            optimizer.zero_grad()

            loss_bc.backward(retain_graph=True)
            w_bc_grad = [weight_list[i].grad for i in range(len(weight_list))]
            w_bc_grad_norm = np.mean([abs(to_numpy(w_pde_grad[i].detach()).flatten()).mean() for i in range(len(w_pde_grad))])
            print(w_bc_grad_norm)         
            optimizer.zero_grad()
            
            loss_ic.backward(retain_graph=True)
            w_ic_grad = [weight_list[i].grad for i in range(len(weight_list))]
            w_ic_grad_norm = np.mean([abs(to_numpy(w_pde_grad[i].detach()).flatten()).mean() for i in range(len(w_pde_grad))])
            print(w_ic_grad_norm)         
            optimizer.zero_grad()

            a = 0.9*a_old + 0.1*w_pde_grad_norm/(w_ic_grad_norm + epsilon)
            b = 0.9*b_old + 0.1*w_pde_grad_norm/(w_bc_grad_norm + epsilon)

            loss = loss_pde + a*loss_bc + b*loss_ic
            
            print(f'epoch {epoch}: loss {loss:.6f} loss_pde {loss_pde:.6f}, loss_bc {loss_bc:.6f}, loss_ic {loss_ic:.6f}')
            loss.backward()
            print(torch.cuda.memory_allocated())

            # a_old = a
            # b_old = b
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    toc = time.time()
    print(f'total training time: {toc-tic}')

if __name__ == '__main__':
    main()


















