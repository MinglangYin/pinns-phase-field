"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - lap(w) = 0
    w - 100*2*(u^2 - u)*(2u-1) + lmd*(d^2(u)/dx^2) = 0, 
    
    x in [0, 1]*[0, 1], t in [0, 1]

    Input: [t, x, y]
    Output: [u, w]
"""

import sys
import os
sys.path.insert(0,'../../../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io

import vtk
import vtk.util.numpy_support as numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from plotting import newfig, savefig

torch.manual_seed(1234)
np.random.seed(1234)

lmd = 1.0e-2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 20))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,15):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(20, 20))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
        self.net.add_module('linear_layer_50', nn.Linear(20, 2))

        self.lmd = torch.nn.Parameter( torch.tensor([1], requires_grad=True, dtype=torch.float32) )

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        y = self.net(x)
        u, w = y[:,0], y[:,1]

        u_g = gradients(u, x)[0]
        u_t, u_x, u_y = u_g[:,0], u_g[:,1], u_g[:,2]
        u_xx = gradients(u_x, x)[0][:,1]
        u_yy = gradients(u_y, x)[0][:,2]

        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:,1], w_g[:,2]
        w_xx = gradients(w_x, x)[0][:,1]
        w_yy = gradients(w_y, x)[0][:,2]

        f = 100*2*u*(u-1)*(2*u-1)

        loss_u = u_t - (w_xx + w_yy)
        loss_w = w - f + self.lmd*(u_xx + u_yy)
        return (loss_u**2).mean() + (loss_w**2).mean()

    def loss_data(self, x, u_data, w_data):
        y = self.net(x)
        u, w = y[:, 0][:, None], y[:, 1][:, None]
        return ((u - u_data)**2).mean() + ((w - w_data)**2).mean()

    # def loss_bc(self, x):
    #     u = self.net(x)
    #     u_x = gradients(u, x)[0][:,1]
    #     return (u_x**2).mean()

    # def loss_ic(self, x, u_init):
    #     u = self.net(x)
    #     return ((u - u_init)**2).mean()

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
    epochs = 5000
    num_x = 9409
    num_t = 10
    lr = 0.001
    dt = 5e-6

    os.makedirs("Predictions", exist_ok=True)

    # The source file
    file_start = 40
    file_end = 49

    for num_file in range(file_start, file_end+1):
        file_u = "./Data/u/output0000" + str(num_file) + ".vtu"
        file_w = "./Data/w/output0000" + str(num_file) + ".vtu"

        reader_u = vtk.vtkXMLUnstructuredGridReader()
        reader_u.SetFileName(file_u)
        reader_u.Update()  # Needed because of GetScalarRange
        
        x_raw = vtk_to_numpy(reader_u.GetOutput().GetPoints().GetData())[:, 0:-1]
        xt_raw = np.concatenate(( (num_file+1)*dt*np.ones((x_raw.shape[0], 1)), x_raw), axis=1)
        u_raw = vtk_to_numpy(reader_u.GetOutput().GetPointData().GetArray("f_8-0"))[:, None]

        reader_w = vtk.vtkXMLUnstructuredGridReader()
        reader_w.SetFileName(file_w)
        reader_w.Update()  # Needed because of GetScalarRange

        w_raw = vtk_to_numpy(reader_w.GetOutput().GetPointData().GetArray("f_8-1"))[:, None]

        if num_file == file_start:
            xt = xt_raw
            u = u_raw
            w = w_raw
        else:
            xt = np.concatenate((xt, xt_raw), axis=0)
            u = np.concatenate((u, u_raw), axis=0)
            w = np.concatenate((w, w_raw), axis=0)
        
    ## sampling
    num_xt_train = 94090
    id_f = np.random.choice(num_x*num_t, num_xt_train, replace=False)

    ## sampling
    x_f = xt[id_f,:]
    u_f = u[id_f,:]
    w_f = w[id_f,:]

    ## tensor
    x_test= torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
    x_f_train = torch.tensor(x_f, requires_grad=True, dtype=torch.float32).to(device)
    u_f_train = torch.tensor(u_f, requires_grad=True, dtype=torch.float32).to(device)
    w_f_train = torch.tensor(w_f, requires_grad=True, dtype=torch.float32).to(device)

    # ##
    x = np.linspace(0, 1, num=num_x)
    y = np.linspace(0, 1, num=num_t)
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid[1:-1,1:-1].flatten()[:,None], x_grid[1:-1,1:-1].flatten()[:,None]), axis=1)

    # ##
    # x_init = np.concatenate((np.zeros((num_x, 1)), x[:, None]), axis=1 )
    # x_b_l = np.concatenate((t_grid[:,0][:,None], x_grid[:,0][:,None]), axis=1)
    # x_b_r = np.concatenate((t_grid[:,-1][:,None], x_grid[:,-1][:,None]), axis=1)
    # x_b = np.concatenate((x_b_l, x_b_r))

    # ## Initial Condition
    # u_init = init_cond(x_init)[:,None]

    # # ## Form data tensor and send
    # x_i_train = torch.tensor(x_init, dtype=torch.float32).to(device)
    # x_b_train = torch.tensor(x_b, requires_grad=True, dtype=torch.float32).to(device)
    # x_f_train = torch.tensor(x_f, requires_grad=True, dtype=torch.float32).to(device)
    # x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    # u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)


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
            loss_data = model.loss_data(x_f_train, u_f_train, w_f_train)
            loss = loss_pde + 100*loss_data
            print(model.lmd)
            print(f'epoch {epoch}: loss {loss:.6f} loss_pde {loss_pde:.6f}, loss_data {loss_data:.6f}')
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')


    u_pred = to_numpy(model(x_test)[:, 0])
    u_pred = u_pred.reshape((num_t, 97, 97))

    for i in range(0, num_t):
        print(f'Saving Step: {i}')
        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(u_pred[i,:,:], interpolation='nearest', cmap='rainbow', origin='lower', aspect="auto")
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./Predictions/u_'+str(i+1000)+'.png')
        plt.close()

if __name__ == '__main__':
    main()


















