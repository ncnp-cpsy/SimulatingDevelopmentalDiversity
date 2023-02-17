import torch
import torch.nn as nn

from model.cell import MaskMaker, MaskedLinear


class InferenceModel(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_time_layer = len(h_dim)

        mask = self.get_mask_matrix()
        self.mean_layer = MaskedLinear(mask.shape[1], mask.shape[0])
        self.mean_layer.set_mask(mask)
        self.std_layer = MaskedLinear(mask.shape[1], mask.shape[0])
        self.std_layer.set_mask(mask)

        self.mean_activation = nn.Tanh()
        # self.std_activation = nn.exp()

        return

    def get_mask_matrix(self):
        mask = MaskMaker.make_mask_matrix(h_dim=self.h_dim,
                                          z_dim=self.z_dim,
                                          mode='pvrnn_inference')
        return mask

    def forward(self, hidden, adaptive_vars=None):
        batch_size = hidden.size(0)

        if type(adaptive_vars) == torch.Tensor:
            # for target generation or error regression (posterior)
            mu_inference = self.mean_activation(
                self.mean_layer(hidden) + \
                adaptive_vars[:, :sum(self.z_dim)]
            )
            sigma_inference = torch.exp(
                self.std_layer(hidden) + \
                adaptive_vars[:, sum(self.z_dim):]
            )
        elif adaptive_vars == None:
            # for free generation
            mu_inference = torch.full(
                (batch_size, sum(self.z_dim)),
                fill_value=float('nan')).to(hidden.device)
            sigma_inference = torch.full(
                (batch_size, sum(self.z_dim)),
                fill_value=float('nan')).to(hidden.device)
        else:
            raise Exception('Error.')

        params = [mu_inference, sigma_inference]
        return params


class InferenceModelNoHidden(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.mean_activation = nn.Tanh()
        # self.std_activation = nn.exp()

    def forward(self, hidden, adaptive_vars=None):
        batch_size = hidden.size(0)
        if type(adaptive_vars) == torch.Tensor:
            # for target generation or error regression (posterior)
            mu_inference = self.mean_activation(
                adaptive_vars[:, :sum(self.z_dim)])
            sigma_inference = torch.exp(
                adaptive_vars[:, sum(self.z_dim):])

        elif adaptive_vars == None:
            # for free generation
            mu_inference = torch.full(
                (batch_size, sum(self.z_dim)),
                fill_value=float('nan')).to(hidden.device)
            sigma_inference = torch.full(
                (batch_size, sum(self.z_dim)),
                fill_value=float('nan')).to(hidden.device)
        else:
            raise Exception('Error.')

        params = [mu_inference, sigma_inference]
        return params

class PriorModel(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(PriorModel, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_time_layer = len(h_dim)

        mask = self.get_mask_matrix()
        self.mean_layer = MaskedLinear(mask.shape[1], mask.shape[0])
        self.mean_layer.set_mask(mask)
        self.std_layer = MaskedLinear(mask.shape[1], mask.shape[0])
        self.std_layer.set_mask(mask)

        self.mean_activation = nn.Tanh()
        # self.std_activation = nn.exp()
        return

    def get_mask_matrix(self):
        mask = MaskMaker.make_mask_matrix(
            h_dim=self.h_dim, z_dim=self.z_dim, mode='pvrnn_inference')
        return mask

    def forward(self, hidden):
        mu_prior = self.mean_activation(self.mean_layer(hidden))
        sigma_prior = torch.exp(self.std_layer(hidden))
            
        params = [mu_prior, sigma_prior]
        return params

class GenerationModelTanh(nn.Module):
    def __init__(self, input_size, x_dim):
        super().__init__()
        self.layer = nn.Linear(input_size, x_dim)
        self.activation = nn.Tanh()

    def forward(self, lowest_hidden, lowest_z=None):
        # combat = torch.cat([lowest_hidden, lowest_z], 1)
        x = self.activation(self.layer(lowest_hidden))
        return x

class GenerationModelGauss(nn.Module):
    def __init__(self, input_size, x_dim):
        super().__init__()

        self.x_dim = x_dim
        self.mean_layer = nn.Linear(input_size, x_dim)
        self.variance_layer = nn.Linear(input_size, x_dim)
        self.mean_activation = nn.Tanh()
    
    def forward(self, lowest_hidden):
        # notice sigma = sqrt of variance
        mean = self.mean_activation(self.mean_layer(lowest_hidden))
        variance = torch.exp(self.variance_layer(lowest_hidden))
        # mean, variance = torch.chunk(self.layer(lowest_hidden), 2, dim=1)
        noise = torch.randn(self.x_dim).to(mean.device)
        out_with_noise = mean + noise * torch.sqrt(variance)

        return mean, variance, noise, out_with_noise

class GenerationModelSoftmax(nn.Module):
    def __init__(self, input_size, x_dim, real_x_dim):
        ''' This generation model is used in Reza-san's source code.
        '''
        super().__init__()

        self.x_dim = x_dim
        self.real_x_dim = real_x_dim
        self.sparse_dim = int(self.x_dim / self.real_x_dim)
        
        self.layer = nn.Linear(input_size, x_dim)
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, lowest_hidden):
        x = self.layer(lowest_hidden)
        for i in range(self.real_x_dim):
            tmp_out = self.activation(
                    x[:, i*self.sparse_dim:i*self.sparse_dim + self.sparse_dim])
            if i == 0: out = tmp_out
            else: out = torch.cat([out, tmp_out], dim=1)
        return out
