"""
XPINNs
"""
import torch
import torch.nn as nn

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

class XPINNs(nn.Module):
    def __init__(self):
        super(XPINNs, self).__init__()
        layer_1, layer_2 = 4, 6
        self.net_1 = Model(layer_1)
        self.net_2 = Model(layer_2)

    def loss_pde(self, x1, x2):
        loss_pde1 = self.net_1.loss_pde(x1)
        loss_pde2 = self.net_2.loss_pde(x2)
        loss = loss_pde1 + loss_pde2
        return loss

    def loss_bc(self, x_l1, x_r1, x_up1, x_dw1, u_l1_real, x_dw2):
        loss_bc1 = self.net_1.loss_bc1(x_l1, x_r1, x_up1, x_dw1, u_l1_real)
        loss_bc2 = self.net_2.loss_bc2(x_dw2)
        loss = loss_bc1 + loss_bc2
        return loss

    def loss_ic(self, x1, u_i1, v_i1, p_i1, phi_i1, x2, u_i2, v_i2, p_i2, phi_i2):
        loss_ic1 = self.net_1.loss_ic1(x1, u_i1, v_i1, p_i1, phi_i1)
        loss_ic2 = self.net_2.loss_ic2(x2, u_i2, v_i2, p_i2, phi_i2)
        loss = loss_ic1 + loss_ic2
        return loss

    def loss_data(self, x_data1, phi_data1, x_data2, phi_data2):
        loss_data1 = self.net_1.loss_data(x_data1, phi_data1)
        loss_data2 = self.net_2.loss_data(x_data2, phi_data2)
        loss = loss_data1 + loss_data2
        return loss

    def loss_interface(self, x_intf):
        y1 = self.net_1(x_intf)
        y2 = self.net_2(x_intf)
        y_avg = (y1 + y2)/2

        K1 = self.net_1.K
        K2 = self.net_2.K
        K_mean = (K1 + K2)/2

        loss = ((y1 - y_avg)**2).mean() + ((y2 - y_avg)**2).mean() + (K1 - K_mean)**2 + (K2 - K_mean)**2

class Model(nn.Module):
    def __init__(self, layer):
        super(Model, self).__init__()
        self.net_w = nn.Sequential()
        self.net_w.add_module('net_w_linear_1', nn.Linear(3, 15))
        self.net_w.add_module('net_w_tanh_1', nn.Tanh())
        for num in range(2,layer):
            self.net_w.add_module('net_w_linear_%d' %(num), nn.Linear(15, 15))
            self.net_w.add_module('net_w_tanh_%d' %(num), nn.Tanh())
        self.net_w.add_module('net_w_linear_50', nn.Linear(15, 1))

        self.net_u = nn.Sequential()
        self.net_u.add_module('net_u_linear_1', nn.Linear(3, 15))
        self.net_u.add_module('net_u_tanh_1', nn.Tanh())
        for num in range(2,layer):
            self.net_u.add_module('net_u_linear_%d' %(num), nn.Linear(15, 15))
            self.net_u.add_module('net_u_tanh_%d' %(num), nn.Tanh())
        self.net_u.add_module('net_u_linear_50', nn.Linear(15, 4))

        self.K = torch.nn.Parameter(torch.tensor([10], requires_grad=True, dtype=torch.double) )

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        output = torch.cat((u, w), dim=1)
        return output

    def loss_pde(self, x):
        y = self.net_u(x)
        w = self.net_w(x)[:,0]

        u, v, p, phi = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        u_g, v_g, p_g, phi_g = gradients(u, x)[0], gradients(v, x)[0], gradients(p, x)[0], gradients(phi, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        v_t, v_x, v_y = v_g[:, 0], v_g[:, 1], v_g[:, 2]
        p_x, p_y = p_g[:, 1], p_g[:, 2]
        phi_t, phi_x, phi_y = phi_g[:, 0], phi_g[:, 1], phi_g[:, 2]

        u_xx, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_y, x)[0][:, 2]
        v_xx, v_yy = gradients(v_x, x)[0][:, 1], gradients(v_y, x)[0][:, 2]
        phi_xx, phi_xy, phi_yy = gradients(phi_x, x)[0][:, 1], gradients(phi_x, x)[0][:, 2], gradients(phi_y, x)[0][:, 2]

        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:, 1], w_g[:, 2]
        w_xx, w_yy = gradients(w_x, x)[0][:, 1], gradients(w_y, x)[0][:, 2]

        ## NS
        res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) + \
                lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y) + mu*(1-phi)*u/(0.001*self.K)
        res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) + \
                lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy) + mu*(1-phi)*v/(0.001*self.K)
        res_incomp = u_x + v_y

        ## PF
        F_g = phi**3 - phi
        res_w = w + h**2*(phi_xx + phi_yy) - gamma*F_g  
        res_pf = phi_t + (u*phi_x + v*phi_y) - tau*(w_xx + w_yy)/h**2

        return (res_pf**2).mean() + (res_incomp**2).mean() + (res_ns_x**2).mean() + (res_ns_y**2).mean() + (res_w**2).mean()

    def loss_data(self, x_data, phi_data):
        y = self.net_u(x_data)
        phi = y[:, 3]
        return ((phi - phi_data)**2).mean()

    def loss_bc1(self, x_l, x_r, x_up, x_dw, u_l_real):
        ## velocity inflow
        y_l, y_r, y_up, y_dw = self.net_u(x_l), self.net_u(x_r), self.net_u(x_up), self.net_u(x_dw)
        w_l, w_r, w_up, w_dw = self.net_w(x_l)[:,0], self.net_w(x_r)[:,0], self.net_w(x_up)[:,0], self.net_w(x_dw)[:,0]
        
        u_l, v_l, phi_l = y_l[:, 0], y_l[:, 1], y_l[:, 3]
        u_r, p_r, phi_r = y_r[:, 0], y_r[:, 2], y_r[:, 3]
        u_up, v_up, phi_up = y_up[:, 0], y_up[:, 1], y_up[:, 3]
        u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3]

        u_r_x = gradients(u_r, x_r)[0][:, 1]

        phi_l_x = gradients(phi_l, x_l)[0][:, 1]
        phi_r_x = gradients(phi_r, x_r)[0][:, 1]
        phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]

        w_l_x = gradients(w_l, x_l)[0][:, 1]
        w_r_x = gradients(w_r, x_r)[0][:, 1]
        w_up_y = gradients(w_up, x_up)[0][:, 2]
        w_dw_y = gradients(w_dw, x_dw)[0][:, 2]

        return ((u_l - u_l_real)**2).mean() + (u_r_x**2).mean() + (u_up**2).mean() + (u_dw**2).mean()\
                + (v_l**2).mean() + (v_up**2).mean() + (v_dw**2).mean() + (p_r**2).mean()\
                + (phi_l_x**2).mean() + (phi_r_x**2).mean() +  (phi_up_y**2).mean() + (phi_dw_y**2).mean() \
                + (w_l_x**2).mean() + (w_r_x**2).mean() + (w_up_y**2).mean() + (w_dw_y**2).mean()

    def loss_bc2(self, x_dw):
        y_dw = self.net_u(x_dw)
        w_dw = self.net_w(x_dw)[:,0]
        u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3]

        phi_dw_y = gradients(phi_dw, x_l)[0][:, 2]
        w_dw_y = gradients(w_l, x_l)[0][:, 2]
        return  (u_dw**2).mean() + (v_dw**2).mean() + (phi_dw_y**2).mean() + (w_dw_y**2).mean()

    def loss_ic(self, x, u_i, v_i, p_i, phi_i):
        y_pred = self.net_u(x)
        u_i_pred, v_i_pred, p_i_pred, phi_i_pred = y_pred[:, 0][:,None], y_pred[:, 1][:,None], y_pred[:, 2][:,None], y_pred[:, 3][:,None]
        return ((u_i_pred-u_i)**2).mean() + ((v_i_pred-v_i)**2).mean() + ((p_i_pred-p_i)**2).mean() + ((phi_i_pred-phi_i)**2).mean() 

