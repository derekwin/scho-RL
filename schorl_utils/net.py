"""

"""
import torch.nn as nn
import torch

def generate_mlpnet(mlp_layers:list, 
        if_batchnorm:bool = False, 
        if_softmax:bool=False,
        if_tahn:bool=False
        ):
    ''' generate a mlpnet
    Args:
        mlp_layers (int): shape of the mlp, [input_node_nums, hidden1_node_nums, ..., hiddenn_node_nums, output_node_nums]
        if_batchnorm (bool): if insert a BatchNorm1d at the first layer.
            when if_batchnorm is True, you must input batch size larger than one.
    Return:
        nn.Sequential()
    '''
    fc = [nn.Flatten()]
    if if_batchnorm:
        fc.append(nn.BatchNorm1d(mlp_layers[0]))
        # batchnorm层限制 batchsize必须大于1
    for i in range(len(mlp_layers)-2):
        fc.append(nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True))
        fc.append(nn.Tanh())
    fc.append(nn.Linear(mlp_layers[-2], mlp_layers[-1], bias=True))
    if if_softmax:
        fc.append(nn.Softmax(dim=1))
    if if_tahn:
        fc.append(nn.Tanh())
    return nn.Sequential(*fc)

class ContinuousPolicyMlp(nn.Module):
    """
    Return:
        use normal distribution to sample action and log_prob
    """
    def __init__(self, mlp_layers:list, if_batchnorm:bool = False) -> None:
        super(ContinuousPolicyMlp, self).__init__()
        fc = [nn.Flatten()]
        if if_batchnorm:
            fc.append(nn.BatchNorm1d(mlp_layers[0]))
            # batchnorm层限制 batchsize必须大于1
        for i in range(len(mlp_layers)-2):
            fc.append(nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True))
            fc.append(nn.Tanh())
        self.mlp = nn.Sequential(*fc)
        self.fc_mean = nn.Linear(mlp_layers[-2], mlp_layers[-1], bias=True)
        self.fc_std = nn.Linear(mlp_layers[-2], mlp_layers[-1], bias=True)

    def forward(self, x):
        x = self.mlp(x)
        mean = self.fc_mean(x)
        std =  nn.functional.softplus(self.fc_std(x))  # 标准差保证为正
        return mean, std


def show_net_structure(net:nn.Module, inputshape:tuple):
    # test batchsize = 5
    testdata = torch.rand((5,*inputshape))
    print(f"The structure of the net:")
    for layer in net:
        testdata = layer(testdata)
        print(layer._get_name(), f"output shape : {testdata.shape}")