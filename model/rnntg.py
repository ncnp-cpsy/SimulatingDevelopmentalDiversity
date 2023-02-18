"""
RNNTG is basis class for MTRNN, SCTRNN and PVRNN.
"""
from abc import abstractmethod

import torch
import torch.nn as nn
torch.set_printoptions(threshold=10000)


class RNNTG(nn.Module):
    def __init__(self):
        super(RNNTG, self).__init__()
        self.device = 'cpu'
        self.data_size = 0
        self.recurrence = None
        self.sort_seq_num_mode = 1

        # Adaptive variables. These variables represents parametric_bias or initial_state.
        self.adaptive_vars_size = 0
        self.adaptive_vars = None
        self.adaptive_vars_ereg = None
        self.init_u_hid = None

        # for saving all units data during forward pass.
        self.all_vars_detached = {}
        self.all_vars_detached_keys = []
        self.all_vars_non_detached = {}
        self.all_vars_non_detached_keys = []

    # _/_/_/ Forward Caliculation _/_/_/

    @abstractmethod
    def _one_step_forward(self):
        pass

    def get_forward_setting(
            self, sequence_number, target,
            max_time_step, closed_threshold):

        if (max_time_step == None) and (type(target) == torch.Tensor):
            max_time_step = target.size(1)
        if type(target) == torch.Tensor:
            target = target.to(self.device)

        '''
        if not (sum(sequence_number >= 0) == sequence_number.size(0) \
                and closed_threshold == max_time_step):
            model.eval()
        else:
            model.train()
        '''

        return sequence_number, max_time_step, closed_threshold, target

    def get_input(self, closed_threshold, step, pre_output=None, target=None):
        """Get input. If closed-loop, hidden[0] are used as input in MTRNN.
        """
        if (step <= closed_threshold):
            observe = target[:, step-1, :] if step == 1 else target[:, step-2, :]
            # print('time ', str(step), ', now open loop')
        elif (step > closed_threshold):
            observe = pre_output
            # print('time ', str(step), ', now closed-loop')

        return observe

    @abstractmethod
    def get_return_output(self):
        pass

    @abstractmethod
    def forward(self, sequence_number, target=None,
                max_time_step=None, closed_threshold=None,
                pb=None, use_saved_u=False):
        pass


        return self.get_return_output()

    # _/_/_/ Initialization of Foword caliculation _/_/_/

    @abstractmethod
    def _initialize(self, sequence_number, use_saved_u=False):
        """This method is used to initializing every forward pass.
        """
        pass

    def set_init_u_hid(self, u_hid):
        if u_hid != None:
            self.init_u_hid = u_hid.detach().clone()
        else:
            self.init_u_hid = None
        # print('initial valued of u_hid unis is saved.\nsaved u: ', self.init_u_hid)
        return

    def get_init_u_hid(self, batch_size=None, use_saved_u=False):
        """
        Basically (other than error regression), initial u_hid and hidden are set zero or randn.
        Using error regression, u_hid are initialized to u_hid at pre-window.
        """
        if (use_saved_u == True) and (self.init_u_hid != None):
            u = self.init_u_hid.detach().clone().to(self.device)
        else:
            u = torch.zeros(batch_size, sum(self.h_dim)).to(self.device)

        # print('initial value is get.', '\ninitial u', u, '\nself saved u', self.init_u_hid)
        return u

    # _/_/_/ Adaptive Variables and Error Regression _/_/_/

    def init_adaptive_vars_ereg(self, step_size):
        self.adaptive_vars_ereg = nn.Parameter(
            torch.randn(
                self.adaptive_vars_size,
                requires_grad=True).to(self.device))
        return

    def set_adaptive_vars_ereg(self, adapt):
        self.adaptive_vars_ereg = nn.Parameter(adapt)

    def get_adaptive_vars_ereg(self):
        return self.adaptive_vars_ereg

    def get_sort_matrix(self, sequence_number):
        batch_size = sequence_number.size(0)
        mat = torch.zeros(batch_size, self.data_size).to(self.device)
        for row_idx in range(batch_size):
            mat[row_idx, sequence_number[row_idx]] = 1
        # print('sort matrix: ', mat)
        return mat

    # _/_/_/ Save Outputs Methods _/_/_/

    @abstractmethod
    def save_vars(self):
        pass

    def clear_vars(self):
        self.all_vars_detached = {}
        self.all_vars_non_detached = {}
        return

    def save_vars_detached(self, values, step, keys=None):
        """ For saving all values during one forward pass.
        """
        batch_size = values[0].size(0)

        if step == 1:
            if keys == None: keys=self.all_vars_detached_keys
            for key, tnsr in zip(keys, values):
                self.all_vars_detached[key] = tnsr.view(batch_size, 1, -1).detach()
        else:
            for key, tnsr in zip(self.all_vars_detached.keys(), values):
                self.all_vars_detached[key] = torch.cat(
                    [self.all_vars_detached[key],
                     tnsr.view(batch_size, 1, -1).detach()], # detached delta error
                    dim=1)
        return

    def save_vars_non_detached(self, values, step, keys=None):
        """ This method is due to loss caliculation, so torch.Tensor is not detached.
        """
        batch_size = values[0].size(0)

        if step == 1:
            if keys == None: keys=self.all_vars_non_detached_keys
            for key, tnsr in zip(keys, values):
                self.all_vars_non_detached[key] = tnsr.view(batch_size, 1, -1)
        else:
            for key, tnsr in zip(self.all_vars_non_detached.keys(), values):
                self.all_vars_non_detached[key] = torch.cat(
                    [self.all_vars_non_detached[key],
                     tnsr.view(batch_size, 1, -1)], # not detached delta error
                    dim=1)
        return
