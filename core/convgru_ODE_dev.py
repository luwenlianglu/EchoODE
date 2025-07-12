import torch
import torch.nn as nn
# import utils
import sys
from torchdiffeq import odeint as odeint

sys.path.append('../')
sys.path.append('./')


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        :param input_dim: int / Number of channels of input tensor.
        :param hidden_dim: int / Number of channels of hidden state.
        :param kernel_size: (int, int) / Size of the convolutional kernel.
        :param bias: bool / Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor / Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        # self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    # def init_hidden(self, batch_size):
    #     return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)

    def forward(self, input_tensor, h_cur, mask=None):
        """
        :param self:
        :param input_tensor: (b, c, h, w) / input is actually the target_model
        :param h_cur: (b, c_hidden, h, w) / current hidden and cell states respectively
        :return: h_next, next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm

        return h_next


def create_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, nonlinear='tanh'):
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    else:
        raise NotImplementedError('There is no named')

    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units, 3, 1, 1, dilation=1))

    for i in range(n_layers):
        layers.append(nonlinear)
        layers.append(nn.Conv2d(n_units, n_units, 3, 1, 1, dilation=1))

    layers.append(nonlinear)
    layers.append(nn.Conv2d(n_units, n_outputs, 3, 1, 1, dilation=1))

    return nn.Sequential(*layers)

class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        # n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict.to(first_point.get_device()),
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict.to(starting_point_enc.get_device()),
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y

class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net):
        """
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim

        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point
		t_local: current time point
		y: value at the current time point
		"""
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        output = self.gradient_net(y)
        return output

    def sample_next_point_from_prior(self, t_local, y):
        """
		t_local: current time point
		y: value at the current time point
		"""
        return self.get_ode_gradient_nn(t_local, y)

class Encoder_z0_ODE_ConvGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3), num_layers=1, batch_first=True,
                 bias=True, return_all_layers=True, z0_diffeq_solver=None, run_backwards=True):

        super(Encoder_z0_ODE_ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.z0_diffeq_solver = z0_diffeq_solver
        self.run_backwards = run_backwards

        ##### By product for visualization
        self.by_product = {}

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

        # last conv layer for generating mu, sigma
        self.z0_dim = hidden_dim[0]
        z = hidden_dim[0]
        self.transform_z0 = nn.Sequential(
            nn.Conv2d(z, z, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(z, z * 2, 1, 1, 0), )

    def forward(self, input_tensor, time_steps):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        assert (input_tensor.size(1) == len(time_steps)), "Sequence length should be same as time_steps"

        last_yi, latent_ys = self.run_ode_conv_gru(
            input_tensor=input_tensor,
            time_steps=time_steps,
            run_backwards=self.run_backwards)

        trans_last_yi = self.transform_z0(last_yi)

        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()

        return mean_z0, std_z0, last_yi

    def run_ode_conv_gru(self, input_tensor, time_steps, run_backwards=True):

        b, t, c, h, w = input_tensor.size()


        # Set initial inputs
        prev_input_tensor = torch.zeros((b, c, h, w))
        if input_tensor.is_cuda:
            prev_input_tensor = prev_input_tensor.to(input_tensor.get_device())

        # Time configuration
        # Run ODE backwards and combine the y(t) estimates using gating
        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]
        latent_ys = []

        time_points_iter = range(0, time_steps.size(-1))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for idx, i in enumerate(time_points_iter):

            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_input_tensor, run_backwards) * (t_i - prev_t)
            assert (not torch.isnan(inc).any())

            # ode_sol = prev_input_tensor + inc
            # ode_sol = torch.stack((prev_input_tensor, ode_sol), dim=1)  # [1, b, 2, c, h, w] => [b, 2, c, h, w]
            # assert (not torch.isnan(ode_sol).any())
            #
            # if torch.mean(ode_sol[:, 0, :] - prev_input_tensor) >= 0.001:
            #     print("Error: first point of the ODE is not equal to initial value")
            #     print(torch.mean(ode_sol[:, :, 0, :] - prev_input_tensor))
            #     exit()

            yi_ode = prev_input_tensor + inc
            xi = input_tensor[:, i, :]

            # only 1 now
            yi = self.cell_list[0](input_tensor=xi,
                                   h_cur=yi_ode)

            # return to iteration
            prev_input_tensor = yi
            prev_t, t_i = time_steps[i], time_steps[i - 1]
            # latent_ys.append(yi)

        # latent_ys = torch.stack(latent_ys, 1)

        return yi, latent_ys

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class forwardDiffFunc(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers=1, n_units=128, nonlinear='tanh'):
        """
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
        super(forwardDiffFunc, self).__init__()
        if nonlinear == 'tanh':
            nonlinear = nn.Tanh()
        else:
            raise NotImplementedError('There is no named')

        layers = []
        layers.append(nn.Conv2d(2*n_inputs, n_units, 3, 1, 1, dilation=1))

        for i in range(n_layers):
            layers.append(nonlinear)
            layers.append(nn.Conv2d(n_units, n_units, 3, 1, 1, dilation=1))

        layers.append(nonlinear)
        layers.append(nn.Conv2d(n_units, n_outputs, 3, 1, 1, dilation=1))

        self.convFunc = nn.Sequential(*layers)
        self.info= 0    #待改正
    def setOverAllInf(self,info):
        self.info=info   #这里也许需要修改设备

    def forward(self, t_local, y, backwards=False):
        x=torch.cat([y,self.info],1)
        return self.convFunc(x)

class forwardDiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5):
        super(forwardDiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
    def setData(self,info):
        self.ode_func.setOverAllInf(info)

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        # n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict.to(first_point.get_device()),
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict.to(starting_point_enc.get_device()),
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


class ConvGRU_ODE_dev(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dilation = 2, timesteps = 4, kernel_size=(3,3), batch_first=True,
                 bias=True, return_all_layers=True):
        super(ConvGRU_ODE_dev, self).__init__()
        self.base_dim = input_dim
        self.ode_dim = hidden_dim
        self.time_dilation = time_dilation
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.build_model()
    def build_model(self):
        #--------------------backward GRU_ODE----------------
        ode_func_netE = create_convnet(n_inputs=self.ode_dim,
                                       n_outputs=self.ode_dim,
                                       n_layers=2,
                                       n_units=self.ode_dim // 2)

        rec_ode_func = ODEFunc(input_dim=self.ode_dim,
                               latent_dim=self.ode_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE)

        z0_diffeq_solver = DiffeqSolver(ode_func=rec_ode_func,
                                        method="euler",
                                        latents=self.ode_dim,
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4)

        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_dim=self.base_dim,
                                            hidden_dim=self.ode_dim,
                                            kernel_size=(3, 3),
                                            num_layers=1,
                                            batch_first=True,
                                            bias=self.bias,
                                            return_all_layers=self.return_all_layers,
                                            z0_diffeq_solver=z0_diffeq_solver,
                                            run_backwards=True)
        #---------------------forward ODE----------------
        ode_func_netD = forwardDiffFunc(n_inputs=self.ode_dim,
                                       n_outputs=self.ode_dim,
                                       n_layers=2,
                                       n_units=self.ode_dim // 2)

        self.diffeq_solver = forwardDiffeqSolver(ode_func_netD,
                                     'dopri5',
                                     self.ode_dim,
                                     odeint_rtol=1e-4,
                                     odeint_atol=1e-5)

    def forward(self, input_tensor, pred_dur = False, input_time_steps=torch.linspace(0, 1, 4), time_steps_to_pred=torch.linspace(0, 1, 4)):
        inputframenum = self.timesteps
        realframenum = (self.timesteps-1)*self.time_dilation+1
        input_time_steps = torch.linspace(0, 1, inputframenum)
        if pred_dur:
            time_steps_to_pred = torch.linspace(0, 1, realframenum)
        else:
            time_steps_to_pred = torch.linspace(0, 1, inputframenum)
        first_point_mu, first_point_std, overall = self.encoder_z0(input_tensor, input_time_steps)
        torch.cuda.manual_seed_all(0)
        eps = torch.randn_like(first_point_mu)
        first_point_enc=eps*first_point_std+first_point_mu
        self.diffeq_solver.setData(overall)
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_pred)
        return sol_y
