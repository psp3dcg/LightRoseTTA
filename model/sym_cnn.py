import torch
import torch.nn as nn
import torch.nn.functional as F

# symmetric CNN layer
class Sym_CNN2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super(Sym_CNN2d, self).__init__()
        '''
        Input:
            - in_channels(int): input channels
            - out_channels(int): output channels
            - kernel_size(int): the size of kernel
        '''
        kernel_size = 3         

        ini_kernel = nn.init.kaiming_normal_(torch.empty([out_channels, in_channels, kernel_size, kernel_size]),mode='fan_out',nonlinearity='relu') 
        ini_bias = nn.init.constant_(torch.empty([1,out_channels,1,1]), val = 0.01) # (B, L, L, d)        
        self.ini_kernel = nn.Parameter(ini_kernel.cuda()) 
        self.ini_bias = nn.Parameter(ini_bias.cuda()) #(B, L, L, d)  

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        '''
        Input:
            - x(tensor): input feature
        Output:
            - output(tensor): output feature
        '''

        # kernel + kernel.T
        mani_kernel = (self.ini_kernel.permute(0,1,3,2)+self.ini_kernel)/2
        output = F.conv2d(x, mani_kernel, padding = int((self.kernel_size-1)/2)) + self.ini_bias
        output = torch.relu(output)

        return output

# symmetric CNN block
class Sym_CNN2d_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size= 3):
        '''
        Input:
            - in_channels(int): input channels
            - out_channels(int): output channels
            - kernel_size(int): the size of kernel
        '''
        super(Sym_CNN2d_Block, self).__init__()
        layer_s = list()

        layer_s.append(Sym_CNN2d(in_channels, hidden_channels, kernel_size))
        layer_s.append(nn.InstanceNorm2d(hidden_channels, affine=True, eps=1e-6))
        layer_s.append(nn.ELU())
        layer_s.append(Sym_CNN2d(hidden_channels, hidden_channels, kernel_size))
        layer_s.append(nn.InstanceNorm2d(hidden_channels, affine=True, eps=1e-6))

        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU()

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

# symmetric CNN network
class Sym_CNN2d_Network(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_blocks=2, kernel_size= 3, p_drop=0.1):
        '''
        Input:
            - in_channels(int): input channels
            - hidden_channels(int): hidden channels
            - out_channels(int): output channels
            - n_blocks(int): the number of blocks
            - kernel_size(int): the size of kernel
        '''
        super(Sym_CNN2d_Network, self).__init__()
        layer_s = list()

        for i_block in range(n_blocks):
            layer_s.append(Sym_CNN2d_Block(in_channels, hidden_channels, kernel_size))

        layer_s.append(Sym_CNN2d(hidden_channels, out_channels, kernel_size))
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