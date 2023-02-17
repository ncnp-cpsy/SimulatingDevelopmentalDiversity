"""Loss Functions.

In this code, normalize loss by only dimension of units for gradient caliculation,
but normalize loss by dimension, num of sequence and sequence length when saving
loss due to comparing other conditions. In Reza-san's code, normalize loss by
dimension and sequence length when saving.

"""

import torch
import torch.nn as nn

import model.computetool as tool


class MTRNNLoss(nn.Module):
    def __init__(self, padding_token=-100.0):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        self.padding_token = padding_token
        return

    def _normalize_loss(self, loss, target=None, mode='grad'):
        if type(target) is torch.Tensor:
            if mode == 'grad':
                factor = target.size(2)
            elif mode == 'save':
                factor = target[target != self.padding_token].numel()

            if factor != 0: loss = loss / factor
        return loss

    def forward(self, output, target):
        loss = self.criterion(
            output[target != self.padding_token],
            target[target != self.padding_token]
        )
        self.loss = self._normalize_loss(loss, target=target, mode='save')
        loss_return = self._normalize_loss(loss, target=target, mode='grad')
        return loss_return


class PVRNNLoss(nn.Module):
    def __init__(self,
                 model,
                 meta_prior,
                 init_meta_prior=None,
                 padding_token=-100.0):
        super().__init__()

        self.model = model
        self.meta_prior = torch.tensor(meta_prior).to(self.model.device)
        if self.model.initial_gaussian_regularizer and (not init_meta_prior is None):
            self.init_meta_prior = torch.tensor(init_meta_prior).to(self.model.device)
        else:
            self.init_meta_prior = self.meta_prior

        self.z_dim = self.model.z_dim
        self.x_dim = torch.tensor(
            self.model.x_dim, dtype=self.meta_prior.dtype).to(self.model.device)

        self.padding_token = padding_token

        self.elbo_loss = torch.zeros(1, requires_grad=True)
        self.kld_loss = torch.zeros(1, requires_grad=True)
        self.kld_loss_non_weighted = torch.zeros(
            (1, len(self.z_dim)), requires_grad=True)
        self.nll_loss = torch.zeros(1, requires_grad=True)

        print('PVRNNLoss Is Initialized. ',
              '\ncriterion: ', self.__class__.__name__,
              '\nmodel: ', self.model.__class__.__name__,
              '\ninitial regularizer: ', self.model.initial_gaussian_regularizer,
              '\nmeta_prior: ', self.meta_prior,
              '\ninit_meta_prior: ', self.init_meta_prior)

        return

    def forward(self, output, target, time_window=None):
        if time_window == None: time_window = target.size(1)

        # nll loss
        nll_loss = self._calc_nll_loss(
            output[target != self.padding_token],
            target[target != self.padding_token])

        # kld loss
        kld_element = self.model.all_vars_non_detached['kld_element'][:,:time_window,:]
        kld_loss, kld_loss_non_weighted, kld_loss_init, kld_loss_other \
            = self._calc_kld_loss(kld_element, target=target)

        # saving
        self._save_loss(
            nll_loss=nll_loss,
            kld_loss=kld_loss,
            kld_loss_non_weighted=kld_loss_non_weighted,
            kld_loss_init=kld_loss_init,
            kld_loss_other=kld_loss_other,
            target=target)

        # normalizing and elbo loss
        nll_loss = self._normalize_nll_loss(nll_loss, target=target, mode='grad')
        kld_loss = self._normalize_kld_loss(
            kld_loss, target=target[target != self.padding_token], mode='grad')
        elbo_loss = nll_loss + kld_loss

        return elbo_loss

    def _normalize_nll_loss(self, loss, target, mode='grad'):
        if type(target) is torch.Tensor:
            if mode == 'grad':
                factor = target.size(2)
            elif mode == 'save':
                factor = target[target != self.padding_token].numel()
            elif mode == 'save_reza':
                factor = target.size(0) * target.size(2)
            if factor != 0: loss = loss / factor
        return loss

    def _normalize_kld_loss(self, loss, target, mode='grad'):
        if type(target) is torch.Tensor:
            if mode == 'grad':
                factor = 1
            elif mode == 'save':
                factor = target[target != self.padding_token].numel() / self.x_dim
            elif mode == 'save_other':
                factor = target[target != self.padding_token].numel() / self.x_dim - target.size(0)
            elif mode == 'save_reza':
                # equivalent to update of reza-san's code
                factor = target.size(0)
            if factor != 0: loss = loss / factor
        return loss

    def _save_loss(self, nll_loss, kld_loss, kld_loss_non_weighted,
                   kld_loss_init, kld_loss_other, target=None):
        self.nll_loss = self._normalize_nll_loss(nll_loss, target=target, mode='save')
        self.kld_loss = self._normalize_kld_loss(kld_loss, target=target, mode='save')
        self.kld_loss_non_weighted \
            = self._normalize_kld_loss(kld_loss_non_weighted, target=target, mode='save')
        self.kld_loss_init \
            = self._normalize_kld_loss(kld_loss_init, target=target, mode='grad')
        self.kld_loss_other \
            = self._normalize_kld_loss(kld_loss_other, target=target, mode='save_other')
        self.elbo_loss = self.nll_loss + self.kld_loss

        return

    def _calc_nll_loss(self, output, target):
        pass

    def _calc_kld_loss(self, kld_element, target, meta_prior=None, init_meta_prior=None):
        meta_prior = self.meta_prior if meta_prior is None else meta_prior
        init_meta_prior = self.init_meta_prior if init_meta_prior is None else init_meta_prior

        # initial (regulalaized) kld loss
        kld_loss_init, kld_loss_init_non_weighted, kld_loss_init_weighted_each_layer \
            = self._calc_kld_loss_weighted(
                kld_element=kld_element[:,0,:].view(target.size(0), 1, sum(self.z_dim)),
                target=target[:,0,:].view(target.size(0), 1, target.size(2)),
                meta_prior=init_meta_prior)

        # other (seqeuntial) kld loss
        kld_loss_other, kld_loss_other_non_weighted, kld_loss_other_weighted_each_layer \
            = self._calc_kld_loss_weighted(
                kld_element=kld_element[:,1:,:].view(target.size(0), -1, sum(self.z_dim)),
                target=target[:,1:,:].view(target.size(0), -1, target.size(2)),
                meta_prior=meta_prior)

        # sum
        kld_loss = kld_loss_init + kld_loss_other
        kld_loss_non_weighted = kld_loss_init_non_weighted + kld_loss_other_non_weighted
        kld_loss_weighted_each_layer = kld_loss_init_weighted_each_layer + kld_loss_other_weighted_each_layer

        '''# debug
        print('meta_prior', self.meta_prior,
              '\ninit_meta_prior', self.init_meta_prior,
              '\nkld_element', kld_element,
              '\nkld_init_non', kld_loss_init_non_weighted,
              '\nkld_other_non', kld_loss_other_non_weighted,
              '\nkld_non', kld_loss_non_weighted,
              '\nkld_init', kld_loss_init,
              '\nkld_other', kld_loss_other,
              '\nkld_loss', kld_loss)
        '''
        # print('kld loss weighted each layer', kld_loss_weighted_each_layer)

        return kld_loss, kld_loss_non_weighted, kld_loss_init, kld_loss_other

    def _calc_kld_loss_weighted(self, kld_element, target, meta_prior):
        kld_loss_non_weighted = self._calc_kld_loss_each_layer(
            kld_element=kld_element, target=target)
        kld_loss_weighted_each_layer = meta_prior * kld_loss_non_weighted
        kld_loss = torch.sum(kld_loss_weighted_each_layer)

        return kld_loss, kld_loss_non_weighted, kld_loss_weighted_each_layer

    def _calc_kld_loss_each_layer(self, kld_element, target):
        # kld element is masked if target is padding out or posterior is not used.
        if target != None:
            mask_padding = target != self.padding_token
            mask_padding = torch.stack(
                [mask_padding[:,:,0] for _ in range(sum(self.z_dim))], axis=2)
        else:
            mask_padding = torch.ones_like(kld_element)
        # mask_posterior = ~ torch.isnan(kld_element)
        mask = mask_padding
        kld_element = mask * kld_element

        '''
        print('kld_element: ', kld_element, 'mask_padding: ', mask_padding,
              'mask_posterior: ', mask_posterior, 'mask', mask
              'product', mask *  kld_element)
        '''

        # calc kld loss for each layer
        for i in range(self.model.num_time_layer):
            kld_loss_add = torch.sum(kld_element[:, :, sum(self.z_dim[:i+1])-self.z_dim[i]:sum(self.z_dim[:i+1])]).view(1)
            kld_loss_add = kld_loss_add / self.z_dim[i]
            kld_loss = kld_loss_add if i == 0 else torch.cat([kld_loss, kld_loss_add], dim=0)

        return kld_loss


class PVRNNTanhLoss(PVRNNLoss):
    def __init__(self, model, meta_prior, init_meta_prior=None, padding_token=-100.0):
        super().__init__(model, meta_prior, init_meta_prior=init_meta_prior,
                         padding_token=padding_token)
        self.criterion = nn.MSELoss(reduction='sum')

    def _calc_nll_loss(self, output, target):
        nll_loss = self.criterion(output, target)
        return nll_loss


class PVRNNSoftmaxLoss(PVRNNLoss):
    def __init__(self, model, meta_prior, init_meta_prior=None, padding_token=-100.0):
        super().__init__(model, meta_prior, init_meta_prior=init_meta_prior,
                         padding_token=padding_token)
        self.real_x_dim = self.model.real_x_dim

    def _normalize_nll_loss(self, loss, target, mode='grad'):
        if type(target) is torch.Tensor:
            if mode == 'grad':
                # factor = target.size(2)
                factor = self.real_x_dim # in reza-san's code using real_x_dim
            elif mode == 'save':
                factor = target[target != self.padding_token].numel()
            elif mode == 'save_reza':
                # factor = target.size(0) * target.size(2)
                factor = target.size(0) * self.real_x_dim # in reza-san's code using real_x_dim
            if factor != 0: loss = loss / factor
        return loss

    def _calc_nll_loss(self, output, target):
        nll_loss = tool.nll_softmax(output, target)
        return nll_loss


