import sys
import torch
import torch.nn as nn
import numpy as np
import time
import random

torch.manual_seed(123456)
np.random.seed(123456)

eta = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 10))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,7):
            self.net.add_module('linear_layer_%d' %(num+1), nn.Linear(10, 10))
            self.net.add_module('tanh_layer_%d' %(num+1), nn.Tanh())
        self.net.add_module('linear_layer_6', nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_x, u_y = u_g[:, 0], u_g[:, 1]
        u_xx = gradients(u_x, x)[0][:, 0]
        u_yy = gradients(u_y, x)[0][:, 1]

        X, Y = x[:, 0], x[:, 1]
        loss = (u_xx + u_yy) - torch.exp(-X)*(X - 2 + Y**3 + 6*Y) ### solving PDE with NN: equ 8, 9
        return (loss**2).mean()

    def loss_bc(self, x, u):
        return ((self.net(x) - u)**2).mean()

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

def AC_2D_solu(x):
    return np.exp(-x[:,0])*(x[:,0]+x[:,1]**3)

def main():
    ## parameters
    device = torch.device(f"cuda:1")
    epochs = 10100
    num_b_train = 10
    num_f_train = 50
    num_test = 50
    lr = 0.001

    x = np.linspace(0, 1, num=num_test)
    y = np.linspace(0, 1, num=num_test)
    x_grid, y_grid = np.meshgrid(x, y)
    x_test = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_b = np.vstack((x_up, x_dw, x_l, x_r))

    u_b = AC_2D_solu(x_b).reshape(num_f_train*4, 1)
    u_f = AC_2D_solu(x_test).reshape(num_f_train*num_f_train, 1)

    id_f = np.random.choice(num_test*num_test, num_f_train)
    id_b = np.random.choice(num_test*4, num_b_train)

    x_f_train = torch.tensor(x_test[id_f], requires_grad=True, dtype=torch.float32).to(device)
    x_b_train = torch.tensor(x_b[id_b], requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)
    u_f = torch.tensor(u_f[id_f], dtype=torch.float32).to(device)
    u_b_train = torch.tensor(u_b[id_b], dtype=torch.float32).to(device)


    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_f_train)
            loss_bc = model.loss_bc(x_b_train, u_b_train)
            loss = loss_pde + loss_bc
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

    ## test
    u_test = to_numpy(model(x_test)).reshape(num_test, num_test)
    x_test = to_numpy(x_test)
    u_sol = to_numpy(AC_2D_solu(x_test)).reshape(num_test, num_test)
    print(f'mean test error is :{((u_test-u_sol)**2).mean()}')

if __name__ == '__main__':
    main()