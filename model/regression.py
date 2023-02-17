import torch


class Regressor(object):
    def __init__(self, model, criterion, optimizer=None,
                 lr=0.001, betas=(0.9, 0.999), ws=30, itrtn=50, pred=1):
        """
        Notice max_pred_step include now step prediction. So, for example,
        if you set two on max_time_step, only now and one step prediction
        is done.
        """
        self.model = model
        self.x_dim = self.model.get_x_dim()

        # parameters of error regression
        self.criterion = criterion
        # self.optimizer = optimizer
        self.max_time_window = ws
        self.max_pred_step = pred
        self.iteration = itrtn
        self.learning_rate = lr
        self.betas = betas

        # for saving all units
        self.save_prediction_step = 1 # from 1 to `self.max_pred_step`
        self.save_postdiction_step = 1 # from 1 to `self.max_time_window`
        self.all_vars_detached = {}
        self.all_vars_detached_keys \
            = [
                var + '_' + str(self.save_prediction_step) + 'step_pred' \
                for var in self.model.all_vars_detached_keys \
            ] + [ \
                var + '_' + str(self.save_postdiction_step) + 'step_post' \
                for var in self.model.all_vars_detached_keys \
            ] + ['all_loss', 'last_loss']

        # debug
        print('Regressor Is Initialized. ',
              '\nmodel: ', self.model.__class__.__name__,
              '\ncriterion: ', self.criterion.__class__.__name__,
              '\nlr: ', self.learning_rate,
              '\nbetas: ', self.betas,
              '\nws: ', self.max_time_window,
              '\nitrtn: ', self.iteration,
              '\npred: ', self.max_pred_step)
        return

    def save_vars_detached(self, values, step, keys=None):
        if keys == None: keys=self.all_vars_detached_keys
        if step == 0:
            for key, tnsr in zip(keys, values):
                # DIM: batch_size x time step x unit dimension
                self.all_vars_detached[key] = tnsr.view(tnsr.size(0), 1, tnsr.size(1)).detach()
        else:
            for key, tnsr in zip(self.all_vars_detached.keys(), values):
                self.all_vars_detached[key] = torch.cat(
                    [self.all_vars_detached[key],
                     tnsr.view(tnsr.size(0), 1, tnsr.size(1)).detach()],
                    dim=1)

    def set_optimizer(self, adapt):
        # betas = (0.5, 0.999) # reza-san's code
        # betas = (0.9, 0.999) # default in torch
        self.optimizer = torch.optim.Adam(
            [adapt], lr=self.learning_rate, betas=self.betas)
        return

    def regress(self, target, use_best_loss=False):
        time_window = 0
        target = target.to(self.model.device)
        batch_size = target.size(0)
        max_time_step = target.size(1)
        best_loss = 0

        # prediction of 1 step ahead
        self.pred_now_step = torch.full(
            (batch_size, max_time_step, self.x_dim), \
            fill_value=float('nan')).to(self.model.device)
        # prediction of `self.max_pred_step` step ahead
        self.pred_last_step = torch.full(
            (batch_size, max_time_step, self.x_dim), \
            fill_value=float('nan')).to(self.model.device)
        # all prediction and postdiction
        self.pred_all = torch.full(
            (batch_size, max_time_step + self.max_pred_step - 1, \
             self.x_dim, self.max_time_window + self.max_pred_step), \
            fill_value=float('nan')).to(self.model.device)
        self.all_adaptive_vars = []

        # initializing of adaptive vars
        self.model.init_adaptive_vars_ereg(step_size=self.max_time_window)

        '''
        # for debug
        print('\ntarget: \n', target,
              '\nadaptive vars: ', self.model.adaptive_vars_ereg)
        '''

        # main of error regression
        for now_step in range(max_time_step):
            self.all_losses = []
            self.set_optimizer(adapt=self.model.get_adaptive_vars_ereg())

            if now_step < self.max_time_window:
                self.model.set_init_u_hid(u_hid=None)
            else:
                self.model.set_init_u_hid(u_hid=u_hid_for_init[:, 1, :])

            # forgetting observation and getting new observation.
            observation = target[:, now_step - time_window:now_step + 1, :]

            for epoch in range(self.iteration):
                # print('\n### now step is', now_step, 'and now iteration is', epoch, '. ###\n')

                # forward pass
                if use_best_loss == True and epoch == self.iteration - 1:
                    self.model.set_adaptive_vars_ereg(adapt=best_adaptive_vars)
                    # print('last iter adapt vars', self.model.get_adaptive_vars_ereg())

                pred = self.predict(max_time_step=time_window + self.max_pred_step,
                                    target=observation, time_window=time_window)

                if epoch == self.iteration - 1 and now_step >= self.max_time_window - 1:
                    u_hid_for_init = self.model.all_vars_detached['u_hid']
                    # print('tmp u_hid is', u_hid_for_init)
                postdiction = pred[:, :time_window, :] # during time window
                prediction = pred[:, time_window:time_window + self.max_pred_step + 1, :] # predicting unknown future observation

                # calculation of prediction error
                loss, loss_pre, loss_post = self.calc_loss(
                    prediction, postdiction, observation, time_window)
                if use_best_loss == True and (loss <= best_loss or epoch == 0):
                    best_loss = loss.item()
                    best_adaptive_vars = self.model.get_adaptive_vars_ereg().clone().detach()
                    # print('iteration', epoch, ': best adapt vars', best_adaptive_vars)

                '''
                # for debug
                print('\npred_step: \t', self.max_pred_step, \
                      '\ntime_window: \t', time_window, \
                      '\niteration: \t', epoch)
                print('\nloss: \t', loss, \
                      '\nloss_pre: \t', loss_pre, \
                      '\nloss_post: \t', loss_post, \
                      '\nall_losses: \t', self.all_losses)
                print('\nprediction used in loss_pre calc: \n', prediction[:, 0, :], \
                      '\nobs used in loss_pre calc: \n', observation[:, time_window, :], \
                      '\npostdiction used in loss_post calc: \n', postdiction, \
                      '\nobs used in loss_post calc: \n', observation[:, :time_window, :])
                print('all variable in model', self.model.all_vars_detached)
                '''

                # updating of adaptive value
                loss.backward()
                if epoch != self.iteration - 1: self.optimizer.step()
                self.optimizer.zero_grad()

            # saving prediction. Mean of Monte Carlo Sampling if using IWL.
            self.pred_now_step[:, now_step, :] \
                = prediction[:, 0, :] if prediction.size(0) == 1 \
                else torch.mean(prediction[:, 0, :], dim=0, keepdim=True)
            self.pred_last_step[:, now_step, :] \
                = prediction[:, -1, :] if prediction.size(0) == 1 \
                else torch.mean(prediction[:, -1, :], dim=0, keepdim=True)
            for i in range(time_window + self.max_pred_step):
                pred_idx = i + self.max_time_window - time_window
                time_idx = i + now_step - time_window
                # print('index', i, pred_idx, time_idx)
                self.pred_all[:, time_idx, :, pred_idx] \
                    = pred[:, i, :] if pred.size(0) == 1 \
                    else torch.mean(pred[:, i, :], dim=0, keepdim=True)

            # saving adaptive vars
            self.save_all_adaptive_vars(self.model.get_adaptive_vars_ereg())

            # saving all variables
            self.save_vars_detached(
                self.get_saving_vars(last_loss=loss.detach(), time_window=time_window),
                step=now_step)

            # updating of time window size
            if time_window < self.max_time_window: time_window += 1

            '''
            # for debug
            for name, param in self.model.named_parameters():
                print(name, param.data)
            print('\npred_step: \t', self.max_pred_step, \
                  '\ntime_window: \t', time_window, \
                  '\npred_all: \t', self.pred_all.size(), \
                  '\nsize of pred: \t: ', pred.size(), \
                  '\npred: \n: ', pred, \
                  '\nprediction: \t', prediction.size(), \
                  '\npostdiction: \t', postdiction.size(), \
                  '\nobservation: \t', observation.size())
            print('\npred: \n', pred, \
                  '\npred_now_step: \n', self.pred_now_step, \
                  '\npred_all: \n', self.pred_all)
            '''
        '''
        print('pred_now_step: \n', self.pred_now_step, \
              '\nall_adaptive_vars: \n', self.all_adaptive_vars, \
              '\npred_all: \n', self.pred_all, \
              '\none step pred of pred_all: \n', self.pred_all[:, :, :, self.max_time_window])
        '''

        return self.pred_now_step

    def predict(self, max_time_step, target, time_window):
        pred = self.model(
            max_time_step=max_time_step,
            target=target,
            closed_threshold=0,
            sequence_number=torch.tensor(-2).view(target.size(0)),
            use_saved_u=True)
        return pred

    def calc_loss(self, prediction, postdiction, observation, time_window):
        loss_pre = self.criterion(
            prediction[:, 0, :].view(prediction.size(0), 1, prediction.size(2)),
            observation[:, time_window, :].view(prediction.size(0), 1, prediction.size(2))
        )

        if time_window == 0:
            loss_post = torch.zeros(1, requires_grad=True).to(self.model.device)
        else:
            loss_post = self.criterion(postdiction, observation[:, :time_window, :])

        loss = loss_post + loss_pre
        self.all_losses.append(loss.item())

        return loss, loss_pre, loss_post

    def save_all_adaptive_vars(self, adapt):
        """
        In now implementation, self.all_adaptive_vars were not saved in results directry.

        DIM (MRTNN, SCTRNN):
        batch_size x target sequential length x self.adaptive_var_size

        DIM (PVRNN):
        batch_size x target sequential length x (self.adaptive_var_size x (window_size + prediction_size))
        """
        batch_size = adapt.size(0)

        if type(self.all_adaptive_vars) == list:
            self.all_adaptive_vars = adapt.view(batch_size, 1, -1).detach().clone()
        else:
            self.all_adaptive_vars = torch.cat(
                [self.all_adaptive_vars,
                 adapt.view(batch_size, 1, -1).detach().clone()
                ], dim=1)
        return

    def get_saving_vars(self, last_loss, time_window):
        """
        Only prediction ahead self.max_pred_step are saved.  The return values
        of self.get_saving_vars were list of dim: batch_size x unit_size.
        """
        now_step_all_vars = []

        # prediction ahead `self.max_pred_step` are saved
        for key, tnsr in self.model.all_vars_detached.items():
            now_step_all_vars.append(tnsr[:, time_window + self.save_prediction_step - 1, :])

        # postdiction back `self.max_pred_step` are saved
        for key, tnsr in self.model.all_vars_detached.items():
            if time_window >= self.save_postdiction_step :
                postdiction = tnsr[:, :time_window, :]
                add = postdiction[:, -self.save_postdiction_step, :]
            else: add = tnsr[:, 0, :]
            now_step_all_vars.append(add)

        # losses are saved
        loss_saving = [
            torch.Tensor(self.all_losses).view(1, self.iteration),
            last_loss.detach().view(1,1)
        ]
        return now_step_all_vars + loss_saving


class RegressorPVRNN(Regressor):
    def __init__(self, model, criterion, optimizer=None,
                 lr=0.001, betas=(0.9, 0.999), ws=30, itrtn=50, pred=1):
        super().__init__(model=model,
                         criterion=criterion, optimizer=optimizer,
                         lr=lr, betas=betas, ws=ws, itrtn=itrtn, pred=pred)
        self.all_vars_detached_keys \
            = [
                var + '_' + str(self.save_prediction_step) + 'step_pred' \
                for var in self.model.all_vars_detached_keys \
            ] + [ \
                var + '_' + str(self.save_postdiction_step) + 'step_post' \
                for var in self.model.all_vars_detached_keys \
            ] + ['all_loss', 'last_loss', 'elbo_loss', 'kld_loss', 'kld_loss_non_weighted', 'nll_loss']
        # + ['kld_loss_non_weighted'] \
        return

    def set_optimizer(self, adapt):
        # self.optimizer = torch.optim.SGD([adapt], lr=self.learning_rate, momentum=0.999)
        # self.optimizer = torch.optim.RMSprop([adapt], lr=self.learning_rate)
        self.optimizer = torch.optim.Adam([adapt], lr=self.learning_rate, betas=self.betas)
        return

    def predict(self, max_time_step, target, time_window):
        pred = self.model(
            max_time_step=max_time_step,
            target=target,
            closed_threshold=time_window,
            sequence_number=torch.tensor(-2).view(target.size(0)),
            use_saved_u=True)
        return pred

    def calc_loss(self, prediction, postdiction, observation, time_window):
        loss_pre = torch.zeros(1, requires_grad=True).to(self.model.device)

        if time_window == 0:
            loss_post = torch.zeros(1, requires_grad=True).to(self.model.device)
        else:
            loss_post = self.criterion(postdiction, observation[:, :time_window, :])

        loss = loss_post + loss_pre
        self.all_losses.append(loss.item())

        return loss, loss_pre, loss_post

    def get_saving_vars(self, last_loss, time_window):
        super_saving_vars = super().get_saving_vars(last_loss=last_loss, time_window=time_window)
        loss_saving_adding = [
            self.criterion.elbo_loss.detach().view(1,1),
            self.criterion.kld_loss.detach().view(1,1),
            self.criterion.kld_loss_non_weighted.detach().view(1, len(self.model.z_dim)),
            self.criterion.nll_loss.detach().view(1,1)
        ]
        return super_saving_vars + loss_saving_adding
