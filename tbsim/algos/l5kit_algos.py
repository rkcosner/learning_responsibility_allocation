from collections import OrderedDict

import torch.nn as nn
from tbsim.models.l5kit_models import RasterizedPlanningModel
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from tbsim.algos.base import PolicyAlgo


class L5TrafficModel(PolicyAlgo):

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=self.algo_config.model_architecture,
            num_input_channels=self.modality_shapes["image"][0], # [C, H, W]
            num_targets=3 * self.algo_config.future_num_frames,  # X, Y, Yaw * number of future states,
            weights_scaling= [1., 1., 1.],
            criterion=nn.MSELoss(reduction="none")
        )

    def process_batch_for_training(self, batch):
        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["prediction_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(L5TrafficModel, self).train_on_batch(batch, epoch, validate=validate)
            pout = self.nets["policy"](batch)
            losses = self.nets["policy"].compute_losses(pout, batch)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

            info["losses"] = TensorUtils.detach(losses)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(PolicyAlgo, self).log_info(info)
        log["Loss"] = info["losses"]["prediction_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict):
        pass
