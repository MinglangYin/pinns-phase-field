"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving Navier-Stokes equation
        u_t + (u*u_x + v*u_y) - nu*(u_xx + u_yy)= -(p_x + P_x) ???
        v_t + (u*v_x + v*v_y) + nu*(v_xx + v_yy) = -p_y

    Input: [x, y, t]
    Output: [u, v, p]
"""

import sys
sys.path.insert(0,'../Utils')
 
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from pyDOE import lhs
from torch.autograd import grad

torch.manual_seed(123456)
np.random.seed(123456)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 50))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for i in range(2, 8):
            self.net.add_module('linear_layer_%d' %(i), nn.Linear(50, 50))
            self.net.add_module('tanh_layer_%d' %(i), nn.Tanh())
        self.net.add_module('linear_layer_8', nn.Linear(50, 3))


    def forward(self, x):
        ## x = [x, t]
        return self.net(x)

    def loss_res(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_x, u_t = u_g[:,:1], u_g[:,1:]
        u_gg = gradients(u_x, x)[0]
        u_xx = u_gg[:,:1]
        loss = u_t + u*u_x - (0.01)/np.pi*u_xx
        return (loss**2).mean()
    
    def loss_icdc(self, x_ic, u_ic, x_bc_l, u_bc_l, x_bc_r, u_bc_r):
        u_pred_ic = self.net(x_ic)
        u_pred_bc_l = self.net(x_bc_l)
        u_pred_bc_r = self.net(x_bc_r)
        loss = ((u_ic-u_pred_ic)**2).mean() +\
                ((u_bc_l-u_pred_bc_l)**2).mean() +\
                ((u_bc_r-u_pred_bc_r)**2).mean()
        return loss
    
    def loss_u(self, x, u_train):
        u_pred = self.net(x)
        return ((u_train - u_pred)**2).mean()


def gradients(outputs, inputs):
    ## it returns a tuple
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
                                create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def main():
    ## parameters
    device = torch.device(f"cpu")
    epochs = 10000
    N_u = 100
    N_f = 10000
    lr = 0.001

    N_train = 5000
    
    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
           
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    x0 = XX[:,0]
    y0 = YY[:,0]
    t0 = TT[:,0]

    

    # Training Data
    idb = np.random.choice(N*T, N_train, replace=False)
    idf = np.random.choice(N*T, N_train, replace=False)
    x_f_train = torch.cat((x[idf,:], y[idf,:], t[idf,:]), dim=1)
    u_f_train = torch.cat((u[idf,:], v[idf,:], p[idf,:]), dim=1)

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss = model.loss_res(x_f_train) + model.loss_u(x_train, u_train)
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
    u_pred = model(x_test)
    u_pred = to_numpy(u_pred)
    U_pred = griddata(x_test, u_pred.flatten(), (X, T), method='cubic')
    print(f'u res is {(np.abs(Exact - U_pred)).mean()}')

    np.savetxt('u_pred.txt', U_pred)
    np.savetxt('exact.txt', Exact)

    ## plotting
    x_train = to_numpy(x_train)
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h, cax=cax)
    ax.plot(x_train[:,1], x_train[:,0], 'kx', label = 'Data (%d points)' % (x_train.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(x.min(), x.max(), 2)[:,None]

    fig1, ax1 = newfig(1.0, 1.1)
    ax1.axis('off')
    ax1.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax1.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax1.set_xlim([-1.1,1.1])
    ax1.set_ylim([-1.1,1.1])

    fig2, ax2 = newfig(1.0, 1.1)
    ax2.axis('off')
    ax2.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax2.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax2.set_xlim([-1.1,1.1])
    ax2.set_ylim([-1.1,1.1])

    fig3, ax3 = newfig(1.0, 1.1)
    ax3.axis('off')
    ax3.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax3.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax3.set_xlim([-1.1,1.1])
    ax3.set_ylim([-1.1,1.1])  

if __name__ == '__main__':
    main()


















