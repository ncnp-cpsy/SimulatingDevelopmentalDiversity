import math
import torch
import numpy as np


def log_mean_exp(x, dim=1, keepdim=True):
    m = torch.max(x, dim=dim, keepdim=keepdim)[0]
    rslt = m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=keepdim))
    return rslt


def nll_bernoulli(theta, x):
    nll = - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))
    return nll


def nll_softmax(out, tar):
    eps = 0.0000000001
    out = out + eps
    tar = tar + eps
    # print(torch.log(tar/out))
    # loss = torch.sum(- tar * torch.log(out)) # general method
    loss = torch.sum(tar * torch.log(tar/out)) # in reza-san's code
    return loss


def nll_gauss_out(mean, variance, target):
    loss = torch.sum((1/2) * torch.log(2 * math.pi * variance) + \
                     (((mean - target) ** 2) / (2 * variance)))
    return loss


def nll_gauss(mean, variance, target):
    sigma = torch.sqrt(variance)
    loss = torch.sum((1/2) * (torch.log(2 * np.pi * (sigma.pow(2)))) + \
                     ((target - mean).pow(2) / (2 * (sigma.pow(2)))))
    return loss


def calc_kld_gauss(params_1, params_2):
    """
    not using std not variance to compute KLD and for one step.
    """
    mean_1, std_1 = params_1
    mean_2, std_2 = params_2

    eps = 0.0000001
    std_1 = std_1 + eps
    std_2 = std_2 + eps
    kld_element =  torch.log(std_2) - torch.log(std_1) + \
                   ((std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (2 * std_2.pow(2))) - (1/2)

    return kld_element


def calc_prob_of_sample_from_gauss(sample, mean, variance):
    """
    the probability of sample ~ Normal(mean, variance) distribution. m
    """
    prob = - 0.5 * torch.log(2 * np.pi * variance) + \
           - ((sample - mean).pow(2) / (2 * variance))

    return prob


def calc_prob_of_sample_from_softmax(output, target):
    eps = 0.0000000001
    out = output + eps
    tar = target + eps
    # loss = tar * torch.log(out) # using general method
    prob = - tar * torch.log(tar/out) # in reza-san's code
    return prob
