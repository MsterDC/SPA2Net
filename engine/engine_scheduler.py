from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    @ Author: Kevin in JLU
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_period: list of target learning rate is reached in warmup_period epochs, gradually, eg:[5,5]
        warmup_node: list of start warmup epoch node, eg:[0,20]
        warmup_params: list of params which use warmup, eg:[['cls','sos'],['sa']]
        optim_params_list: list of all params, eg:['cls_weight', 'cls_bias', 'sos_weight', 'sos_bias', 'sa_weight', 'sa_bias', 'bb_weight', 'bb_bias']
    """

    def __init__(self, optimizer, warmup_period, warmup_node, warmup_params, optim_params_list, multiplier=1.0):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_period = warmup_period  # [5,5]
        self.warmup_node = warmup_node  # [0,20]
        self.stage = [(node, node+period) for node, period in zip(warmup_node, warmup_period)]  #[(0,5),(20,25)]
        self.warmup_params = warmup_params  # [['cls_weight', 'cls_bias', 'sos_weight', 'sos_bias'], ['sa_weight', 'sa_bias']]
        self.optim_params = optim_params_list  # ['cls_weight', 'cls_bias', 'sos_weight', 'sos_bias', 'sa_weight', 'sa_bias', 'bb_weight', 'bb_bias']
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """ Variable 'self.last_epoch' is equal to 0 at the beginning of training.
        If current epoch is in the warmup period, parameters' learning rates are all set to 0 except 'warmup_params'.
        :return: updated learning rate => list
        """
        for idx, (start_epoch, end_epoch) in enumerate(self.stage):
            if start_epoch <= self.last_epoch <= end_epoch:  # while current epoch in any period of warmup
                new_lr = []
                for p_name, base_lr in zip(self.optim_params, self.base_lrs):
                    exist_flag = False
                    for w_name in self.warmup_params[idx]:
                        if p_name == w_name:
                            if self.multiplier == 1.0:
                                updated_lr = base_lr * ((self.last_epoch + 1 - start_epoch) / self.warmup_period[idx] + 1)
                            else:
                                updated_lr = base_lr * ((self.multiplier - 1.) * (self.last_epoch + 1 - start_epoch) / self.warmup_period[idx] + 1. + 1)
                            new_lr.append(updated_lr)
                            exist_flag = True
                    # parameters' learning rates are all set to 0 except 'warmup_params'.
                    if exist_flag is False:
                        new_lr.append(float(0))
                return new_lr
        # while current epoch not in any period of warmup
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        :param epoch: equal to 0 at the beginning of training.
        :return: learning rate calculated by 'get_lr()' function.
        """
        return super(GradualWarmupScheduler, self).step(epoch)

    def update_optimizer(self, optimizer, epoch):
        """ Update scheduler when optimizer has been changed.
        :param optimizer: Wrapped optimizer.
        :param epoch: current epoch
        """
        # First, update field 'initial_lr' of param_groups.
        for group in optimizer.param_groups:
            group.update({'initial_lr': group['lr']})
        print("[Tips:] Initial_lr has been updated.")
        # Second, update field self.base_lrs of the scheduler.
        super(GradualWarmupScheduler, self).__init__(optimizer, epoch)
        print("[Tips:] base_lrs of warmup scheduler has been updated.")