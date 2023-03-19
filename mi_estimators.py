import numpy as np

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.nn.parameter import Parameter
SMALL = 1e-08


class UpperBound(nn.Module, ABC):

    @abstractmethod
    def update(self, y_samples):

        raise NotImplementedError


class CLUBForCategorical(nn.Module):  # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''

    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()

        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU(),
                                                 nn.Linear(hidden_size, label_num))

    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  #[sample_size, label_num]

        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')
        sample_size, label_num = logits.shape

        logits_extend = logits.unsqueeze(1).repeat(1, sample_size,
                                                   1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size,
                                                   1)  # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = -nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num), labels_extend.reshape(-1, ), reduction='none')

        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return -nn.functional.cross_entropy(logits, labels)

    def learning_loss(self, inputs, labels):
        return -self.loglikeli(inputs, labels)

class UpperWithPosterior(UpperBound):
    """
    后验分布 p(y|x) 均假设为高斯分布
    """

    def __init__(self, embedding_dim=768, hidden_dim=500, tag_dim=128, device=None):
        super(UpperWithPosterior, self).__init__()
        self.device = device
        # u
        self.p_mu = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(hidden_dim, tag_dim))
        # log(σ**2)
        self.p_log_var = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden_dim, tag_dim),
            # nn.Tanh()
        )

    # 返回 u , log(σ**2)
    def get_mu_logvar(self, embeds):
        mean = self.p_mu(embeds)
        log_var = self.p_log_var(embeds)

        return mean, log_var

    def loglikeli(self, y_samples, mu, log_var):
        # [batch, seq_len, dim]
        return (-0.5 * (mu - y_samples)**2 / log_var.exp() + log_var +
                torch.log(math.pi)).sum(dim=1).mean(dim=0)

    # 从正态分布中 sample 样本
    def get_sample_from_param_batch(self, mean, log_var, sample_size):
        bsz, seqlen, tag_dim = mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to(self.device)

        z = z * torch.exp(0.5 * log_var).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)

        # [batch * sample_size, seq_len, tag_dim]
        return z

    @abstractmethod
    def update(self, y_samples):
        raise NotImplementedError


class VIB(UpperWithPosterior):
    """
    Deep Variational Information Bottleneck
    """

    # 表示该高斯分布与 N（0，1）之间的KL散度
    def update(self, x_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)

        return 1. / 2. * (mu**2 + logvar.exp() - 1. - logvar).mean()


class CLUB(UpperWithPosterior):
    """
    CLUB: Mutual Information Contrastive Learning Upper Bound
    """

    def mi_est_sample(self, y_samples, mu, log_var):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size, )).long()

        positive = -(mu - y_samples)**2 / 2. / log_var.exp()
        negative = -(mu - y_samples[random_index])**2 / 2. / log_var.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def update(self, x_samples):
        """
        f(x_sample) = u, var -> sampling -> y_sample
        return mi_est_sample(x_sample, y_sample)
        :param x_samples:
        :return:
        """
        mu, log_var = self.get_mu_logvar(x_samples)
        y_samples = self.get_sample_from_param_batch(mu, log_var, 1).squeeze(0)

        return self.mi_est_sample(y_samples, mu, log_var)


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2), nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2), nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim), nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = -(mu - y_samples)**2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = -((y_samples_1 - prediction_1)**2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)

class vCLUB(UpperBound):
    """
     vCLUB: Variational Mutual Information Contrastive Learning Upper Bound
    """

    def __init__(self):
        super(vCLUB, self).__init__()

    def mi_est_sample(self, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size, )).long()

        return self.mi_est(y_samples, y_samples[random_index])

    def mi_est(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)
        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """
        positive = torch.zeros_like(y_samples)

        negative = -(y_samples - y_n_samples)**2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mse(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)
        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """

        return (y_samples - y_n_samples)**2 / 2

    def consine(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)
        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """
        return torch.cosine_similarity(y_samples, y_n_samples, dim=-1)

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, y_samples, y_n_samples):

        return self.mi_est(y_samples, y_n_samples)


class InfoNCE(nn.Module):

    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(InfoNCE, self).__init__()
        if hidden_size is None:
            hidden_size = (x_dim + y_dim) // 2
        # self.down_p = nn.Sequential(nn.Linear(x_dim, y_dim), nn.ReLU())
        # self.bilinear = nn.Bilinear(y_dim, y_dim, 1, False)
        self.w = Parameter(nn.init.kaiming_uniform_(torch.randn(1, x_dim, y_dim)))
        # self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size), nn.ReLU(),
        #                             nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        # T0 = self.bilinear(self.down_p(x_samples), y_samples)
        # T1 = self.bilinear(self.down_p(x_tile), y_tile)
        T0 = torch.einsum("ij,mjk,ik->im", [x_samples, self.w, y_samples])
        T1 = torch.einsum("bij,mjk,bik->bim", [x_tile, self.w, y_tile])
        # T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


def kl_div(param1, param2):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    """
    # u, log(std**2)
    mean1, log_cov1 = param1
    mean2, log_cov2 = param2
    cov1 = log_cov1.exp()
    cov2 = log_cov2.exp()
    bsz, seqlen, tag_dim = mean1.shape
    var_len = tag_dim * seqlen

    cov2_inv = 1 / cov2
    mean_diff = mean1 - mean2

    mean_diff = mean_diff.view(bsz, -1)
    cov1 = cov1.view(bsz, -1)
    cov2 = cov2.view(bsz, -1)
    cov2_inv = cov2_inv.view(bsz, -1)

    temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
    KL = 0.5 * (torch.sum(torch.log(cov2), dim=1) - torch.sum(torch.log(cov1), dim=1) - var_len +
                torch.sum(cov2_inv * cov1, dim=1) +
                torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))

    return KL.mean()


def kl_norm(mu, log_var):
    """
    :param mu: u
    :param log_var: log(std**2)
    :return:
        D_kl(N(u, std**2), N(0, 1))
    """

    return -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1).mean()
