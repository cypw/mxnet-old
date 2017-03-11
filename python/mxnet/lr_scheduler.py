"""
learning rate scheduler, which adaptive changes the learning rate based on the
progress
"""
import logging

class LRScheduler(object):
    """Base class of a learning rate scheduler"""
    def __init__(self):
        """
        base_lr : float
            the initial learning rate
        """
        self.base_lr = 0.01

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        The training progress is presented by `num_update`, which can be roughly
        viewed as the number of minibatches executed so far. Its value is
        non-decreasing, and increases at most by one.

        The exact value is the upper bound of the number of updates applied to
        a weight/index

        See more details in https://github.com/dmlc/mxnet/issues/625

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class FactorScheduler(LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(floor(n/step))

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0
        self.first_call = True

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if self.first_call:
            self.first_call = False
            logging.info("- Init[%s|%s]: Start learning rate from %0.5e",
                         str(num_update).rjust(8), str(int(self.count+self.step)).rjust(8), self.base_lr)

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr <= self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%s|%s]: now learning rate arrived at %0.5e, will not "
                             "change in the future",
                             str(num_update).rjust(8), ('Inf').rjust(8), self.base_lr)
            else:
                logging.info("Update[%s|%s]: Change learning rate to %0.5e",
                             str(num_update).rjust(8), str(int(self.count+self.step)).rjust(8), self.base_lr)
        return self.base_lr

class MultiFactorScheduler(LRScheduler):
    """Reduce learning rate in factor at steps specified in a list

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(sum((step/n)<=1)) # step is an array

    Parameters
    ----------
    step: list of int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, step, factor=1):
        super(MultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.first_call = True

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if self.first_call:
            self.first_call = False
            logging.info("- Init[%s|%s]: Start learning rate from %0.5e",
                         str(num_update).rjust(8), str(self.step[self.cur_step_ind]).rjust(8), self.base_lr)

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.cur_step_ind < len(self.step):
            while num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                if self.cur_step_ind >= len(self.step):
                    logging.info("Update[%s|%s]: now learning rate arrived at %0.5e, will not "
                                 "change in the future",
                                 str(num_update).rjust(8), ('Inf').rjust(8), self.base_lr)
                    return self.base_lr
                else:
                    logging.info("Update[%s|%s]: Change learning rate to %0.5e",
                                 str(num_update).rjust(8), str(self.step[self.cur_step_ind]).rjust(8), self.base_lr)
        return self.base_lr
