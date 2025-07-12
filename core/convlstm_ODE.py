import torch
from torch import nn
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding, bias=True):
        super(ConvLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding, bias=bias)
        self._init_weights()

    def forward(self, x, prev_state):
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return (hidden, cell)

    def _init_weights(self):
        torch.nn.init.xavier_normal_(self.gates.weight.data, 0.02)
class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, padding=1, num_layers=1, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.layers = self.build_layers()

    def build_layers(self):
        layers = []
        for i in range(self.num_layers):
            curr_input_size = self.input_size if i == 0 else self.hidden_size
            layers.append(ConvLSTMCell(curr_input_size, self.hidden_size, self.kernel_size, self.padding, self.bias))

        return nn.Sequential(*layers)


    def forward(self, x, hidden_state=(None, None)):
        seq_len = x.size(1)
        current_input = x
        output = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                hidden_state = self.init_hidden(x)

            for t in range(seq_len):
                hidden_state = layer(current_input[:, t, :, :, :], hidden_state)
                output.append(hidden_state[0])

            output = torch.stack(output, dim=1)
            current_input = output

        return output

    def init_hidden(self, x):
        batch_size = x.size()[0]
        spatial_size = x.size()[3:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        if 0:#torch.cuda.is_available():
            return (Variable(torch.zeros(state_size)).cuda(), Variable(torch.zeros(state_size)).cuda())
        else:
            return (Variable(torch.zeros(state_size)), Variable(torch.zeros(state_size)))

def get_net(args):
    return ConvLSTM_ODE(latent_dim = args.latent_dim,
                   ode_func_layers = args.ode_func_layers,
                   ode_func_units = args.ode_func_units,
                   input_dim = args.input_dim,
                   decoder_units = args.decoder_units)
class ODEFunc(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        mid_channels = int(in_channels/2)
        super(ODEFunc, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)
    def forward(self, t_local, y, backwards=False):
        return self.layers(y)

class DiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		# assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		# assert(pred_y.size()[0] == n_traj_samples)
		# assert(pred_y.size()[1] == n_traj)
		return pred_y

class ConvLSTM_ODE(nn.Module):
    """Class for standalone ODE-RNN model. Makes predictions forward in time."""
    def __init__(self, in_channels, latent_channels):
        super(ConvLSTM_ODE, self).__init__()

        ode_func = ODEFunc(in_channels=latent_channels, mid_channels=latent_channels)

        self.ode_solver = DiffeqSolver(ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)


        self.convLstmCell = ConvLSTMCell(input_size = in_channels, hidden_size = latent_channels, kernel_size = (3,3), padding = 1)

        self.latent_channels = latent_channels

        self.sigma_fn = nn.Softplus()

    def forward(self, data, extrap_time=float('inf'), use_sampling=False):
        batch_size, n_time_steps, input_channel, input_height, input_width = data.size()
        time_steps = torch.linspace(0, 1, n_time_steps+1)

        prev_hidden = torch.zeros((batch_size, self.latent_channels, input_height, input_width))
        cell_ode = torch.zeros((batch_size, self.latent_channels, input_height, input_width))

        if data.is_cuda:
            prev_hidden = prev_hidden.to(data.get_device())
            cell_ode = cell_ode.to(data.get_device())

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        outputs = []
        for i in range(1, len(time_steps)):

            # Make one step.
            if time_steps[i] - time_steps[i - 1] < 0:
                inc = self.ode_solver.ode_func(time_steps[i - 1], prev_hidden)

                ode_sol = prev_hidden + inc * (time_steps[i] - time_steps[i - 1])
                ode_sol = torch.stack((prev_hidden, ode_sol), 1)
            # Several steps.
            else:
                num_intermediate_steps = max(2, ((time_steps[i] - time_steps[i - 1])/minimum_step).int())

                time_points = torch.linspace(time_steps[i - 1], time_steps[i],
                                             num_intermediate_steps)
                ode_sol = self.ode_solver(prev_hidden, torch.tensor([time_steps[i - 1], time_steps[i]]).to(data.get_device()))

            hidden_ode = ode_sol[-1]

            x_i = data[:, i-1]
            prev_hidden, cell_ode = self.convLstmCell(x_i, (hidden_ode, cell_ode))
            outputs.append(prev_hidden)

        outputs = torch.stack(outputs, 1)
        return outputs

if __name__ == "__main__":
    # x = torch.randn(2, 5, 3, 112, 112)
    # net = ConvLSTM_ODE(in_channels = 3, latent_channels=3)
    # y=net(x)

    net = ConvLSTM(input_size=3, hidden_size=16, kernel_size=(3, 3),
                            batch_first=True)
    x = torch.randn(2, 5, 3, 480, 480)
    y = net(x)


