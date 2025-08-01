import torch.nn as nn

from core.tcn2d import TemporalConvNet2D
from core.tcn import TemporalConvNet
from core.tcn2dhw import TemporalConvNet2DHW
from core.lstm import LSTM
from core.convlstm import ConvLSTM
from torchdiffeq import odeint_adjoint as odeint

class BaseTemporalModel(nn.Module):

    def __init__(self, args):
        super(BaseTemporalModel, self).__init__()

        self.args = args
        self.timesteps = args.timesteps
        self.mode = args.mode
        self.sequence_model_type = args.sequence_model
        self.sequence_stacked_models = args.sequence_stacked_models

    def get_sequence_model(self, args):
        """
            Stacks a number of inplace_abn f to model the temporal dimension of the sequence
        """
        layers = []
        for i in range(args.sequence_stacked_models):
            args.current_skip_level = args.num_downs
            layers.append(self.build_sequence_model(args))

        return nn.Sequential(*layers)

    def get_skip_sequence_models(self, args):
        """
            Returns an array of inplace_abn f to model the temporal dimension of the skip connections
        """
        skip_connection_models = []
        for i in range(args.num_downs - 1):
            args.current_skip_level = i
            skip_connection_models.append(self.build_sequence_model(args))

        return nn.Sequential(*skip_connection_models)

    def build_sequence_model(self, args):
        """
            Constructs a module f to model the temporal dimension of the sequence
        """
        if 'convlstm' in args.sequence_model:
            number_channels = self.get_number_channels(args)
            return ConvLSTM(input_size=number_channels, hidden_size=number_channels, kernel_size=(3, 3),
                            batch_first=True)
        elif 'lstm' in args.sequence_model:
            lstm_size = args.ngf * 8 * int(args.resize[0] / 2 ** args.num_downs) * int(args.resize[1] / 2 ** args.num_downs)
            return LSTM(input_size=lstm_size, hidden_size=lstm_size, batch_first=True,
                        bidirectional=args.lstm_bidirectional, lstm_initial_state=args.lstm_initial_state)
        elif 'tcn2dhw' in args.sequence_model:
            number_channels = self.get_number_channels(args)
            number_channels = number_channels * self.timesteps
            return TemporalConvNet2DHW(num_inputs=number_channels, num_channels=[number_channels] * args.num_levels_tcn,
                                       kernel_size=args.tcn_kernel_size)
        elif 'tcn2d' in args.sequence_model:
            number_channels = self.get_number_channels(args)
            return TemporalConvNet2D(num_inputs=number_channels, num_channels=[number_channels] * args.num_levels_tcn,
                                     kernel_size=args.tcn_kernel_size)
        elif 'tcn' in args.sequence_model:
            number_channels = [self.timesteps] * args.num_levels_tcn
            return TemporalConvNet(num_inputs=self.timesteps, num_channels=number_channels, kernel_size=1)
        else:
            raise Exception('Unsupported sequence model')

    def temporal_forward(self, x, sequence_model, hidden_init=None):
        batch_size, C, H, W = int(x.size(0) / self.timesteps), int(x.size(1)), int(x.size(2)), int(x.size(3))

        if 'convlstm' in self.sequence_model_type:
            x = x.view(batch_size, self.timesteps, C, H, W)
        elif 'tcn2dhw' in self.sequence_model_type:
            x = x.view(batch_size, self.timesteps * C, H, W)
        elif 'tcn2d' in self.sequence_model_type:
            x = x.view(batch_size, self.timesteps, C, -1)
        else:
            x = x.view(batch_size, self.timesteps, -1)  # tcn, lstm, gru (etc)

        x = sequence_model(x)
        x = x.contiguous().view(batch_size * self.timesteps, C, H, W)

        return x

    def skip_connection_temporal_forward(self, skip_connections):
        new_skip_connections = []
        for i in range(len(skip_connections)):
            x = skip_connections[i]
            skip_sequence_model = self.skip_sequence_models._modules[str(i)]
            batch_size, C, H, W = int(x.size(0) / self.timesteps), int(x.size(1)), int(x.size(2)), int(x.size(3))

            if 'convlstm' in self.sequence_model_type:
                x = x.view(batch_size, self.timesteps, C, H, W)
            elif 'tcn2dhw' in self.sequence_model_type:
                x = x.view(batch_size, self.timesteps * C, H, W)
            elif 'tcn2d' in self.sequence_model_type:
                x.view(batch_size, self.timesteps, C, -1)
            else:
                x.view(batch_size, self.timesteps, -1)  # tcn, lstm, gru (etc)

            x = skip_sequence_model(x)
            x = x.contiguous().view(batch_size * self.timesteps, C, H, W)

            new_skip_connections.append(x)

        return new_skip_connections

    def get_number_channels(self, args):
        return args.ngf * pow(2, args.current_skip_level) if args.current_skip_level <= 3 else args.ngf * 8

    def get_train_parameters(self, lr):
        segmentation_params = [{'params': self.parameters(), 'lr': lr}]

        return segmentation_params

    def remove_time_reshape(self, x):
        if 'sequence' in self.mode:
            batch_size, timesteps, C, H, W = x.size()
            x = x.view(batch_size * timesteps, C, H, W)

        return x

    def add_time_reshape(self, x):
        if 'sequence' in self.mode:
            x = x.view(-1, self.timesteps, x.size(1), x.size(2), x.size(3))

        return x

