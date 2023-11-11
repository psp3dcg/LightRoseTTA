import torch
import torch.nn as nn

# original resblock
class ResBlock2D(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        '''
        Input:
            - n_c(int): in channel and out channel for CNN kernel 
            - kernel(int): kernel size (3x3)
            - dilation(int): use dilation convolution or not
            - p_drop(float): dropout ratio
        '''
        super(ResBlock2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        layer_s = list()
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # convolution
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU()

    def _get_same_padding(self, kernel, dilation):
        '''
        Input:
            - kernel(int): kernel size
            - dilation(int): use dilation convolution or not
        Output:
            - output_kernel(int): kernel size with dilation
        '''
        output_kernel = (kernel + (kernel - 1) * (dilation - 1) - 1) // 2
        return output_kernel

    def forward(self, x):
        '''
        Input:
            - x(tensor): input feature
        Output:
            - output(tensor): output feature
        '''
        out = self.layer(x)
        output = self.final_activation(x + out)
        return output

# pre-activation bottleneck resblock
class ResBlock2D_bottleneck(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        '''
        Input:
            - n_c(int): in channel and out channel for CNN kernel 
            - kernel=3(int): kernel size (3x3)
            - dilation(int): use dilation convolution or not
            - p_drop(float): dropout ratio
        '''
        super(ResBlock2D_bottleneck, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        n_b = n_c // 2 # bottleneck channel
        
        layer_s = list()
        # pre-activation
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        # project down to n_b
        layer_s.append(nn.Conv2d(n_c, n_b, 1, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        # convolution
        layer_s.append(nn.Conv2d(n_b, n_b, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # project up
        layer_s.append(nn.Conv2d(n_b, n_c, 1, bias=False))

        self.layer = nn.Sequential(*layer_s)

    def _get_same_padding(self, kernel, dilation):
        '''
        Input:
            - kernel(int): kernel size
            - dilation(int): use dilation convolution or not
        Output:
            - output_kernel(int): kernel size with dilation
        '''
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        '''
        Input:
            - x(tensor): input feature
        Output:
            - output(tensor): output feature
        '''
        out = self.layer(x)
        output = x + out
        return output

# residual network
class ResidualNetwork(nn.Module):
    def __init__(self, n_block, n_feat_in, n_feat_block, n_feat_out, 
                 dilation=[1,2,4,8], block_type='orig', p_drop=0.15):
        '''
        Input:
            - n_block(int): the number of blocks
            - n_feat_in(int): the dimension of input feature
            - n_feat_block(int): the dimension of hidden feature
            - n_feat_out(int): the dimension of output feature
            - dilation(list): the dilation number list 
            - block_type(str): the type of block
            - p_drop(float): dropout ratio
        '''
        super(ResidualNetwork, self).__init__()


        layer_s = list()
        # project to n_feat_block
        if n_feat_in != n_feat_block:
            layer_s.append(nn.Conv2d(n_feat_in, n_feat_block, 1, bias=False))
            if block_type =='orig': # should acitivate input
                layer_s.append(nn.InstanceNorm2d(n_feat_block, affine=True, eps=1e-6))
                layer_s.append(nn.ELU())

        # add resblocks
        for i_block in range(n_block):
            d = dilation[i_block%len(dilation)]
            if block_type == 'orig':
                res_block = ResBlock2D(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            else:
                res_block = ResBlock2D_bottleneck(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            layer_s.append(res_block)

        if n_feat_out != n_feat_block:
            # project to n_feat_out
            layer_s.append(nn.Conv2d(n_feat_block, n_feat_out, 1))
        
        self.layer = nn.Sequential(*layer_s)
    
    def forward(self, x):
        '''
        Input:
            - x(tensor): input feature
        Output:
            - output(tensor): output feature
        '''
        output = self.layer(x)
        return output

