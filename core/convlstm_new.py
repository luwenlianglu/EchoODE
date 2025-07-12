import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        初始化卷积 LSTM 单元。

        参数:
        ----------
        input_dim: int
            输入张量的通道数。
        hidden_dim: int
            隐藏状态的通道数。
        kernel_size: (int, int)
            卷积核的大小。
        bias: bool
            是否添加偏置项。
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # 计算填充大小以保持输入和输出尺寸一致
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 定义卷积层，输入是输入维度加上隐藏维度，输出是4倍的隐藏维度（对应i, f, o, g）
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 沿着通道轴进行拼接
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        # 将输出分割成四个部分，分别对应输入门、遗忘门、输出门和候选单元状态
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新单元状态
        c_next = f * c_cur + i * g
        # 更新隐藏状态
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # 初始化隐藏状态和单元状态为零
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    卷积 LSTM 层。

    参数:
    ----------
    input_dim: 输入通道数
    hidden_dim: 隐藏通道数
    kernel_size: 卷积核大小
    num_layers: LSTM 层的数量
    batch_first: 批次是否在第一维
    bias: 卷积中是否有偏置项
    return_all_layers: 是否返回所有层的计算结果

    输入:
    ------
    一个形状为 B, T, C, H, W 或者 T, B, C, H, W 的张量

    输出:
    ------
    元组包含两个列表（长度为 num_layers 或者长度为 1 如果 return_all_layers 为 False）：
    0 - layer_output_list 是长度为 T 的每个输出的列表
    1 - last_state_list 是最后的状态列表，其中每个元素是一个 (h, c) 对应隐藏状态和记忆状态

    示例:
    >>> x = torch.rand((32, 10, 64, 128, 128))
    >>> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
    >>> _, last_states = convlstm(x)
    >>> h = last_states[0][0]  # 0 表示层索引，0 表示 h 索引
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # 检查 kernel_size 的一致性
        self._check_kernel_size_consistency(kernel_size)

        # 确保 kernel_size 和 hidden_dim 的长度与层数一致
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('不一致的列表长度。')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 创建 ConvLSTMCell 列表
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        前向传播函数。

        参数:
        ----------
        input_tensor: 输入张量，形状为 (t, b, c, h, w) 或者 (b, t, c, h, w)
        hidden_state: 初始隐藏状态，默认为 None

        返回:
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # 改变输入张量的顺序，如果 batch_first 为 False
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # 实现状态化的 ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # 初始化隐藏状态
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # 在每个时间步上更新状态
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # 将输出堆叠起来
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            # 如果不需要返回所有层，则只返回最后一层的输出和状态
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            # 初始化每一层的隐藏状态
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` 必须是 tuple 或者 list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    x = torch.rand((32, 10, 64, 128, 128))
    convlstm = ConvLSTM(64, 16, (3,3), 2, True, True, False)
    _, last_states = convlstm(x)
    h = last_states[0][0]  # 0 表示层索引，0 表示 h 索引
