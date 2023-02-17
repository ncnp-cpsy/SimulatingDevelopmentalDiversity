"""
Adaptive values are parameter list. list reflect time-scale layer,
each variable's size is [data_size, z_dim[layer], time_step].
Lowest (index = 0) layer is fast context. In detail, see _generation
method.
"""

import torch
import torch.nn as nn

from model.cell import MaskMaker
from model.component import InferenceModel, InferenceModelNoHidden, \
    PriorModel, GenerationModelTanh, GenerationModelSoftmax
from model.mtrnn import MTRNN
from model.computetool import calc_kld_gauss


class PVRNNTanh(MTRNN):
    def __init__(self, x_dim, h_dim, z_dim,
                 time_scale, data_size, max_time_step, device,
                 num_iw=1, num_mc=1,
                 initial_gaussian_regularizer=True,
                 use_hidden_for_posterior=True,
                 use_bottom_up_signal=True):

        # hyper-parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.use_bottom_up_signal = use_bottom_up_signal
        self.initial_gaussian_regularizer = initial_gaussian_regularizer
        self.max_time_step = max_time_step
        self.num_iw = num_iw
        self.num_mc = num_mc

        super().__init__(
            h_dim=h_dim, time_scale=time_scale,
            data_size=data_size, device=device)

        # module of PVRNN (recurrece component is initialized in MTRNN)
        self.inference = InferenceModel(h_dim, z_dim) if use_hidden_for_posterior else \
            InferenceModelNoHidden(h_dim, z_dim)
        self.prior = PriorModel(h_dim, z_dim)
        self.generation = GenerationModelTanh(h_dim[0], x_dim)

        # for initial state and adaptive value and error regression
        self.adaptive_vars_size = sum(self.z_dim) * 2
        adaptive_vars = torch.randn(
            data_size, max_time_step, sum(self.z_dim)*2).to(self.device)
        adaptive_vars[:, :, :sum(self.z_dim)] \
            = 0.0 * adaptive_vars[:, :, :sum(self.z_dim)]
        adaptive_vars[:, :, sum(self.z_dim):] \
            = 0.0 * adaptive_vars[:, :, sum(self.z_dim):]
        self.adaptive_vars = nn.Parameter(adaptive_vars).to(self.device)

        # for saving
        self.all_vars_detached_keys = [
            'all_x',
            'all_d', 'u_hid', 'all_z',
            'all_mu', 'all_sigma',
            'all_mu_inference', 'all_sigma_inference',
            'all_mu_prior', 'all_sigma_prior',
            'kld_element']
        self.all_vars_non_detached_keys = [
            'all_x', 'all_z',
            'all_mu_inference', 'all_sigma_inference',
            'all_mu_prior', 'all_sigma_prior',
            'kld_element']

    def get_mask_matrix(self):
        mode = 'pvrnn_recurrence' if self.use_bottom_up_signal == True \
               else 'pvrnn_recurrence_off-bottom-up'
        mask = MaskMaker.make_mask_matrix(
            h_dim=self.h_dim, z_dim=self.z_dim, mode=mode)
        return mask

    def _initialize(self, use_saved_u=False, batch_size=0):
        u_hid = self.get_init_u_hid(
            batch_size=batch_size, use_saved_u=use_saved_u)
        hidden = self.recurrence.activation(u_hid)
        return hidden, u_hid

    def _one_step_forward(self, hidden, u_hid, sequence_number, step):
        # inference (encoder, posterior)
        adaptive_vars_now = self.get_adaptive_vars(sequence_number, step)
        params_inference = self.inference(hidden, adaptive_vars_now)

        # prior
        if self.initial_gaussian_regularizer == True and step == 1:
            params_prior = [torch.zeros(sequence_number.size(0),
                                        sum(self.z_dim)).to(self.device),
                            torch.ones(sequence_number.size(0),
                                       sum(self.z_dim)).to(self.device)]
        else:
            params_prior = self.prior(hidden)

        # reparametarization tric
        params = self._check_mode(
            params_inference, params_prior, sequence_number)
        z = self._reparameterized_sample(params)

        # recurrence
        combat = torch.cat((hidden, z), 1)
        hidden, u_hid = self.recurrence(hidden=combat, u_hid=u_hid)

        # generation
        x = self.generation(hidden[:, :self.h_dim[0]])

        # kld element
        kld_element = self._calc_kld_loss_one_step(
            params_inference, params_prior)

        # saving
        self.save_vars(
            x=x, hidden=hidden, u_hid=u_hid, z=z,
            params=params,
            params_inference=params_inference,
            params_prior=params_prior,
            kld_element=kld_element, step=step)

        return x, hidden, u_hid

    def save_vars(self, x, hidden, u_hid, z,
                  params, params_inference, params_prior,
                  kld_element, step):
        self.save_vars_detached(
            [x, hidden, u_hid, z] + \
            params + params_inference + params_prior + \
            [kld_element], step=step)
        self.save_vars_non_detached(
            [x, z] + \
            params_inference + params_prior + \
            [kld_element], step=step)
        return

    def get_forward_setting(
            self, sequence_number, target,
            max_time_step, closed_threshold):

        if (max_time_step == None) and (type(target) == torch.Tensor):
            max_time_step = target.size(1)
        if type(target) == torch.Tensor:
            target = target.to(self.device)
        if closed_threshold == None:
            closed_threshold = max_time_step

        target, sequence_number \
            = self.resample_target(target=target, sequence_number=sequence_number)

        return sequence_number, max_time_step, closed_threshold, target

    def resample_target(self, target, sequence_number=None):
        """
        importance weighted sampleing. this method is also accessed from loss function.
        """
        if self.num_iw != 1 or self.num_mc != 1:
            print('using iwl, targets were resampled.')
            if target is not None:
                target = target.repeat(self.num_mc * self.num_iw, 1, 1)
                #print('post of resampled target', target.size())
            if sequence_number is not None:
                sequence_number = sequence_number.repeat(self.num_mc * self.num_iw)
                #print('post of seq num', sequence_number)

        return target, sequence_number

    def forward(self,
                sequence_number, target=None,
                max_time_step=None, closed_threshold=None,
                use_saved_u=False):
        """Main forward method.
        learning -> sequence_number > 0 and closed_threshold = max_time_step or None
        (If training, please set closed_threshold=None)
        target generation -> sequence_number > 0 and closed_threshold > 0
        free generation -> sequence_number == -1 and closed_threshold <= 0
        error regression -> sequence_number == -2
        """

        # setting
        self.clear_vars()
        sequence_number, max_time_step, closed_threshold, target \
            = self.get_forward_setting(
                sequence_number=sequence_number,
                target=target,
                max_time_step=max_time_step,
                closed_threshold=closed_threshold)
        batch_size = sequence_number.size(0)

        # progressive learning
        self.change_mask_setting(sequence_number, closed_threshold, max_time_step)

        # initializing
        hidden, u_hid = self._initialize(
            use_saved_u=use_saved_u, batch_size=batch_size)
        if self.sort_seq_num_mode == 1 \
           and sum(sequence_number >= 0) == batch_size:
            self.sort_matrix = self.get_sort_matrix(sequence_number)

        '''
        print('target: ', target)
        print('initialized based on sequence_number : ', sequence_number,
              'initialized u_hid: ', u_hid,
              'initialized hidden', hidden)
        '''

        # forward pass from time step 1 to max_time_step
        for step in range(1, max_time_step + 1):
            if step > closed_threshold and (
                    sum(sequence_number >= 0) == batch_size or # target generation
                    sum(sequence_number == -2) == batch_size # error regression
            ):
                sequence_number = -1 * torch.ones(batch_size)
                '''
                print('shifting from posterior to prior.',
                      'now step: ', step, 'and seq num: ', sequence_number,
                      'closed_thresh: ', closed_threshold)
                '''

            # one step forward pass
            x, hidden, u_hid = self._one_step_forward(
                hidden=hidden, u_hid=u_hid,
                sequence_number=sequence_number, step=step)

            '''
            print('x: ', x,
                  '\nhidden: ', hidden,
                  '\nu_hide: ', u_hid)
            '''

        output = self.get_return_output()
        return output

    def get_return_output(self):
        if self.num_iw == 1 and self.num_mc == 1:
            output = self.all_vars_non_detached['all_x']
        else:
            # this is dummy return. this retrun is not used backprop
            output = self.compress_samples(tnsr=self.all_vars_detached['all_x'])
        return output

    def compress_samples(self, tnsr, use_mean=False):
        num_sample = self.num_iw * self.num_mc
        batch_size_original = int(tnsr.size(0) / num_sample)
        if use_mean:
            idx_list_sampling = [[j + batch_size_original * i \
                                  for i in range(num_sample)] \
                                 for j in range(batch_size_original)]
            # print(idx_list_sampling)
            tnsr_return = torch.stack(
                [torch.mean(tnsr[idx_list_sampling[i], :, :], dim=0, keepdim=False) \
                 for i in range(batch_size_original)
                ], axis=0)
        else:
            tnsr_return = tnsr[:batch_size_original,:,:]

        return tnsr_return

    def get_x_dim(self):
        return self.x_dim

    def init_adaptive_vars_ereg(self, step_size):
        self.adaptive_vars_ereg = nn.Parameter(torch.randn(
            step_size, self.adaptive_vars_size, requires_grad=True).to(self.device))

    def get_adaptive_vars(self, sequence_number, step):
        if sum(sequence_number >= 0) == sequence_number.size(0):
            adaptive_vars = self.sort_matrix.mm(self.adaptive_vars[:, step - 1, :])
        elif sum(sequence_number == -1) == sequence_number.size(0):
            adaptive_vars = None
        elif sum(sequence_number == -2) == sequence_number.size(0):
            adaptive_vars = self.adaptive_vars_ereg[step - 1, :].view(
                -1, self.adaptive_vars_size)
        return adaptive_vars















    # _/_/_/ latent space traversal _/_/_/

    def _sample_other_z(self, params=None, batch_size=1, generator_other_z='zeros'):
        if generator_other_z == 'zeros':
            z = torch.zeros(batch_size, sum(self.z_dim)).to(self.device)
        elif generator_other_z == 'randn':
            z = torch.randn(batch_size, sum(self.z_dim)).to(self.device)
        elif generator_other_z == 'prior':
            z = self._reparameterized_sample(params=params, rand_off=False)
        return z

    def _sample_target_z(self,
                         params_prior,
                         target_idx,
                         generator_mu,
                         generator_sigma,
                         generator_noise,
                         fixed_value_mu=None,
                         fixed_value_sigma=None,
                         fixed_value_noise=None,
                         batch_size=1):

        if generator_mu == 'prior':
            mu = params_prior[0][:, target_idx]
        elif generator_mu == 'fix' or generator_mu == 'traversal':
            mu = torch.full((batch_size, 1), fill_value=fixed_value_mu)
        else:
            raise Exception('pepepepep')

        if generator_sigma == 'prior':
            sigma = params_prior[1][:, target_idx]
        elif generator_sigma == 'fix' or generator_sigma == 'traversal':
            sigma = torch.full((batch_size, 1), fill_value=fixed_value_sigma)
        else:
            raise Exception('pepepepep')

        if generator_noise == 'fix' or generator_noise == 'traversal' :
            noise = torch.full((batch_size, 1), fill_value=fixed_value_noise).to(self.device)
        elif generator_noise == 'rand':
            noise = torch.randn(batch_size, 1).to(self.device)
        else:
            raise Exception('pepepepep')

        z = noise * sigma + mu

        '''
        print('pre replacing of mu and sigma: ',
              params_prior[0][:, target_idx], params_prior[1][:, target_idx],
              'post replacing of mu and sigma: ', mu, sigma )
        '''

        return z

    def _sample_one_step(
            self, hidden, u_hid, step,
            target_idx,
            generator_mu,
            generator_sigma,
            generator_noise,
            generator_other_z='zeros',
            fixed_value_mu=None,
            fixed_value_sigma=None,
            fixed_value_noise=None,
            using_target_values=True):
        batch_size=hidden.size(0)

        # generate prior params z for other than target idx
        params_prior = self.prior(hidden)
        z = self._sample_other_z(
            params=params_prior,
            batch_size=batch_size,
            generator_other_z=generator_other_z)

        # generate prior params and z for target idx
        if using_target_values:
            # print('z of target_idx was replaced.')
            z[:, target_idx] = self._sample_target_z(
                params_prior=params_prior,
                target_idx=target_idx,
                generator_mu=generator_mu,
                generator_sigma=generator_sigma,
                generator_noise=generator_noise,
                fixed_value_mu=fixed_value_mu,
                fixed_value_sigma=fixed_value_sigma,
                fixed_value_noise=fixed_value_noise,
                batch_size=batch_size
            )

        # recurrence
        combat = torch.cat((hidden, z), 1)
        hidden, u_hid = self.recurrence(hidden=combat, u_hid=u_hid)

        # generation
        x = self.generation(hidden[:, :self.h_dim[0]])

        # saving
        self.save_vars_detached([x, hidden, u_hid, z], step=step)
        self.save_vars_non_detached([x], step=step)

        return x, hidden, u_hid

    def _sample(
            self,
            target_idx,
            generator_mu,
            generator_sigma,
            generator_noise,
            generator_other_z='zeros',
            fixed_value_mu=None,
            fixed_value_sigma=None,
            fixed_value_noise=None,
            batch_size=1,
            max_time_step=None,
            only_initial_step=False):

        # setting for traversal
        sample_range = 256
        sample_value_start = -2
        sample_value_end = 2
        traversal_value = sample_value_start

        # initializing
        self.clear_vars()
        hidden, u_hid = self._initialize(
            use_saved_u=False, batch_size=batch_size)

        for step in range(1, max_time_step):
            # set traversal_value
            if generator_mu == 'traversal': fixed_value_mu = traversal_value
            if generator_sigma == 'traversal': fixed_value_sigma = traversal_value
            if generator_noise == 'traversal': fixed_value_noise = traversal_value
            using_target_values = True if not only_initial_step or (only_initial_step and step == 1) else False
            # print("step: ", step, " using_target_values", using_target_values)

            # one step forward
            x, hidden, u_hid = self._sample_one_step(
                hidden=hidden, u_hid=u_hid, step=step,
                target_idx=target_idx,
                generator_mu=generator_mu,
                generator_sigma=generator_sigma,
                generator_noise=generator_noise,
                fixed_value_mu=fixed_value_mu,
                fixed_value_sigma=fixed_value_sigma,
                fixed_value_noise=fixed_value_noise,
                generator_other_z=generator_other_z,
                using_target_values=using_target_values
            )

            # update of fixed_value
            if traversal_value <= sample_value_start: sign = 1
            elif traversal_value >= sample_value_end: sign = -1
            traversal_value += sign * (sample_value_end - sample_value_start) / sample_range

        return self.get_return_output()

    def latent_space_traversal(
            self, mode,
            target_idx,
            generator_other_z='prior',
            fixed_value=None,
            batch_size=1,
            max_time_step=None):
        """
        Parameters
        ----------
        mode : str
            'constant_z', 'constant_sigma', 'constant_noise', 'traversal_z', 'traversal_noise', 'constant_z_only_initial'
        target_idx : int
            index of units
        generator_mu : str
            'prior', 'fix' or 'traversal'
        generator_sigma : str
            'prior', 'fix' or 'traversal'
        generator_noise : str
            'fix' or 'traversal'
        generator_other_z : str
            'zeros', 'randn', or 'prior'
        """

        if max_time_step is None: max_time_step = self.max_time_step * 2

        all_vars_detached_keys_tmp = self.all_vars_detached_keys
        all_vars_non_detached_keys_tmp = self.all_vars_non_detached_keys
        self.all_vars_detached_keys = ['all_x', 'all_d', 'u_hid', 'all_z']
        self.all_vars_non_detached_keys = ['all_x']

        if mode == 'constant_z':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='fix',
                generator_sigma='fix',
                generator_noise='rand',
                generator_other_z=generator_other_z,
                fixed_value_mu=fixed_value,
                fixed_value_sigma=0,
                fixed_value_noise=None, # random generator
                batch_size=batch_size,
                max_time_step=max_time_step)
        elif mode == 'constant_sigma':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='prior',
                generator_sigma='fix',
                generator_noise='rand',
                generator_other_z=generator_other_z,
                fixed_value_mu=None,
                fixed_value_sigma=fixed_value,
                fixed_value_noise=None,
                batch_size=batch_size,
                max_time_step=max_time_step)
        elif mode == 'constant_noise':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='prior',
                generator_sigma='prior',
                generator_noise='fix',
                generator_other_z=generator_other_z,
                fixed_value_mu=None,
                fixed_value_sigma=None,
                fixed_value_noise=fixed_value,
                batch_size=batch_size,
                max_time_step=max_time_step)
        elif mode == 'traversal_z':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='traversal',
                generator_sigma='fix',
                generator_noise='fix',
                generator_other_z=generator_other_z,
                fixed_value_mu=None,
                fixed_value_sigma=0,
                fixed_value_noise=0,
                batch_size=batch_size,
                max_time_step=max_time_step)
        elif mode == 'traversal_noise':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='prior',
                generator_sigma='prior',
                generator_noise='traversal',
                generator_other_z=generator_other_z,
                fixed_value_mu=None,
                fixed_value_sigma=None,
                fixed_value_noise=None,
                batch_size=batch_size,
                max_time_step=max_time_step)
        elif mode == 'constant_z_only_initial':
            output = self._sample(
                target_idx=target_idx,
                generator_mu='fix',
                generator_sigma='fix',
                generator_noise='rand',
                generator_other_z=generator_other_z,
                fixed_value_mu=fixed_value,
                fixed_value_sigma=0,
                fixed_value_noise=None, # random generator
                batch_size=batch_size,
                max_time_step=max_time_step,
                only_initial_step=True)

        else:
            raise Exception('apapapap')

        self.all_vars_detached_keys = all_vars_detached_keys_tmp
        self.all_vars_non_detached_keys = all_vars_non_detached_keys_tmp

        return output

    # _/_/_/ iroiro _/_/_/
    def _check_mode(self, params_inference, params_prior, sequence_number):
        if sum(sequence_number == -2) == sequence_number.size(0):
            params = params_inference
            # print('using posterior (error regression)')
        elif sum(sequence_number == -1) == sequence_number.size(0):
            params = params_prior
            # print('using prior')
        elif sum(sequence_number >= 0) == sequence_number.size(0):
            params = params_inference
            # print('using posterior (target generation)')

        return params

    def _reparameterized_sample(self, params, rand_off=False):
        mean, std = params
        noise = torch.randn(std.size(0), std.size(1)).to(self.device)

        if rand_off == True:
            mask = torch.zeros(std.size(0), std.size(1)).to(self.device)
            ''' below is rand-off in target layer
                idx_start = 0
                for i, num_z in enumerate(self.z_dim):
                    if rand_off[i]:
                    mask[:, idx_start:idx_start + num_z] \
                        = torch.zeros(std.size(0), num_z).to(self.device)
                idx_start += num_z
            '''
            noise = mask * noise
            print('noise set to zeros. mask is ', mask)

        z = noise * std + mean

        #print('mean', mean,  '\nstd: ', std, '\nnoise: ', noise, '\nz: ', z)

        return z

    def _calc_kld_loss_one_step(self, params_1, params_2):
        if torch.sum(torch.isnan(params_1[0])) == params_1[0].numel():
            # when free generation (not using posterior), kld_element is nan or zero.
            kld_element = torch.zeros_like(params_1[0]).to(params_1[0].device)
            # kld_element = torch.full_like(params_1[0], fill_value=float('nan')).to(params_1[0].device)
        else:
            kld_element = calc_kld_gauss(params_1, params_2)
        return kld_element

    def change_mask_setting(self, sequence_number, closed_threshold, max_time_step):
        pass


class PVRNNSoftmax(PVRNNTanh):
    def __init__(self, x_dim, real_x_dim, h_dim, z_dim,
                 time_scale, data_size, max_time_step, device,
                 num_iw=1, num_mc=1,
                 initial_gaussian_regularizer=True,
                 use_hidden_for_posterior=True,
                 use_bottom_up_signal=True):

        super().__init__(x_dim, h_dim, z_dim,
                         time_scale, data_size, max_time_step, device,
                         num_iw=num_iw, num_mc = num_mc,
                         initial_gaussian_regularizer=initial_gaussian_regularizer,
                         use_hidden_for_posterior=use_hidden_for_posterior,
                         use_bottom_up_signal=use_bottom_up_signal)

        self.real_x_dim = real_x_dim
        self.generation = GenerationModelSoftmax(
            input_size = h_dim[0],
            x_dim = x_dim,
            real_x_dim = real_x_dim)

        return
