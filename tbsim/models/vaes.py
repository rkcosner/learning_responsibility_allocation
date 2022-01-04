"""Variants of Conditional Variational Autoencoder (C-VAE)"""
from collections import OrderedDict

import torch
import torch.nn as nn
from tbsim.utils.loss_utils import KLD_0_1_loss, KLD_gaussian_loss
from tbsim.utils.torch_utils import reparameterize
import tbsim.utils.tensor_utils as TensorUtils


class Prior(nn.Module):
    def __init__(self, latent_dim, input_dim=None, device=None):
        """
        A generic prior class
        Args:
            latent_dim: dimension of the latent code (e.g., mu, logvar)
            input_dim: (Optional) dimension of the input feature vector, for conditional prior
            device:
        """
        super(Prior, self).__init__()
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self._net = None
        self._device = device

    def forward(self, inputs: torch.Tensor = None):
        """
        Get a batch of prior parameters.

        Args:
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            params (dict): A dictionary of prior parameters with the same batch size as the inputs (1 if inputs is None)
        """
        raise NotImplementedError

    def sample(self, n: int, inputs: torch.Tensor = None):
        """
        Take samples with the prior distribution

        Args:
            n (int): number of samples to take
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [input_batch_size, n, latent_dim]
        """
        prior_params = self.forward(inputs=inputs)
        return self.sample_with_parameters(prior_params, n=n)

    @staticmethod
    def sample_with_parameters(params: dict, n: int):
        """
        Take samples using given a batch of distribution parameters, e.g., mean and logvar of a unit Gaussian

        Args:
            params (dict): a batch of distribution parameters
            n (int): number of samples to take

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [param_batch_size, n, latent_dim]
        """
        raise NotImplementedError

    def kl_loss(self, posterior_params: dict, inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """
        raise NotImplementedError

    @property
    def posterior_param_shapes(self) -> dict:
        """
        Shape of the posterior parameters

        Returns:
            shapes (dict): a dictionary of parameter shapes
        """
        raise NotImplementedError


class FixedGaussianPrior(Prior):
    """An unassuming unit Gaussian Prior"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(FixedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = nn.ParameterDict({
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False)
        })

    def forward(self, inputs: torch.Tensor = None):
        """
        Get a batch of prior parameters.

        Args:
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            params (dict): A dictionary of prior parameters with the same batch size as the inputs (1 if inputs is None)
        """

        batch_size = 1 if inputs is None else inputs.shape[0]
        params = TensorUtils.unsqueeze_expand_at(self._params, size=batch_size, dim=0)
        return params

    @staticmethod
    def sample_with_parameters(params, n: int):
        """
        Take samples using given a batch of distribution parameters, e.g., mean and logvar of a unit Gaussian

        Args:
            params (dict): a batch of distribution parameters
            n (int): number of samples to take

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [param_batch_size, n, latent_dim]
        """

        batch_size = params["mu"].shape[0]
        params_tiled = TensorUtils.repeat_by_expand_at(params, repeats=n, dim=0)
        samples = reparameterize(params_tiled["mu"], params_tiled["logvar"])
        samples = TensorUtils.reshape_dimensions(samples, begin_axis=0, end_axis=1, target_dims=(batch_size, n))
        return samples

    def kl_loss(self, posterior_params, inputs=None):
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """

        assert posterior_params["mu"].shape[1] == self._latent_dim
        assert posterior_params["logvar"].shape[1] == self._latent_dim
        return KLD_0_1_loss(
            mu=posterior_params["mu"],
            logvar=posterior_params["logvar"]
        )

    @property
    def posterior_param_shapes(self) -> OrderedDict:
        return OrderedDict(mu=(self._latent_dim,), logvar=(self._latent_dim,))


class LearnedGaussianPrior(FixedGaussianPrior):
    """A Gaussian prior with learnable parameters"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(LearnedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = nn.ParameterDict({
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True)
        })

    def kl_loss(self, posterior_params, inputs=None):
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """

        assert posterior_params["mu"].shape[1] == self._latent_dim
        assert posterior_params["logvar"].shape[1] == self._latent_dim

        batch_size = posterior_params["mu"].shape[0]
        prior_params = TensorUtils.unsqueeze_expand_at(self._params, size=batch_size, dim=0)
        return KLD_gaussian_loss(
            mu_1=posterior_params["mu"],
            logvar_1=posterior_params["logvar"],
            mu_2=prior_params["mu"],
            logvar_2=prior_params["logvar"]
        )


class CVAE(nn.Module):
    def __init__(
            self,
            q_encoder: nn.Module,
            c_encoder: nn.Module,
            decoder: nn.Module,
            prior: Prior,
            target_criterion: nn.Module
    ):
        """
        A basic Conditional Variational Autoencoder Network (C-VAE)
        Args:
            q_encoder (nn.Module): a model that encodes data (x) and condition inputs (x_c) to posterior (q) parameters
            c_encoder (nn.Module): a model that encodes condition inputs (x_c) into condition feature (c)
            decoder (nn.Module): a model that decodes latent (z) and condition feature (c) to data (x')
            prior (nn.Module): a model containing information about distribution prior (kl-loss, prior params, etc.)
            target_criterion (nn.Module): a loss function for target reconstruction
        """
        super(CVAE, self).__init__()
        self.q_encoder = q_encoder
        self.c_encoder = c_encoder
        self.decoder = decoder
        self.prior = prior
        self.target_criterion = target_criterion

    def sample(self, condition_inputs: dict, n: int):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict): condition inputs - a dictionary of named tensors (x_c)
            n (int): number of samples to draw

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        c = self.c_encoder(condition_inputs)  # [B, ...]
        z = self.prior.sample(n=n, inputs=c)  # z of shape [B (from c), N, ...]
        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * N, ...]
        x_out = self.decoder(latents=z_samples, conditions=c_samples)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c.shape[0], n))
        return x_out

    def forward(self, inputs, condition_inputs):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict): encoder inputs (x)
            condition_inputs (dict): condition inputs - a dictionary of named tensors (x_c)

        Returns:
            dictionary of batched samples (x')
        """
        q_params = self.q_encoder(inputs=inputs, condition_inputs=condition_inputs)
        z = self.prior.sample_with_parameters(q_params, n=1).squeeze(dim=1)
        c = self.c_encoder(condition_inputs)  # [B, ...]
        x_out = self.decoder(latents=z, conditions=c)
        return {"x_recons": x_out, "q_params": q_params, "z": z, "c": c}

    def compute_losses(self, outputs, targets, target_weights=None, kl_weight: float = 1.0):
        """
        Compute VAE losses

        Args:
            outputs (dict): outputs of the self.forward() call
            targets (dict): reconstruction targets (x')
            target_weights (dict): (Optional) a dictionary of floats for weighing loss for reconstructing individual
                                    elements in the target, must be the same shape as the target.
            kl_weight (float): weighting factor for the KL divergence loss

        Returns:
            a dictionary of loss values
        """
        recon_loss = 0
        if target_weights is None:
            target_weights = dict()
        x_recons = outputs["x_recons"]
        for k, v in x_recons.items():
            w = target_weights[k] if k in target_weights else torch.ones_like(targets[k])
            element_loss = self.target_criterion(x_recons[k], targets[k]) * w
            recon_loss += torch.mean(element_loss)  # TODO: sum up last dimension instead of averaging all?

        kld_loss = self.prior.kl_loss(outputs["q_params"], inputs=outputs["c"]) * kl_weight

        return {"recon_loss": recon_loss, "kl_loss": kld_loss}


def main():
    import tbsim.models.l5kit_models as l5m
    prior = FixedGaussianPrior(latent_dim=16)

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        num_input_channels=3,
        visual_feature_dim=128
    )

    q_encoder = l5m.PosteriorEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        output_shapes=OrderedDict(mu=(16,), logvar=(16,))
    )
    c_encoder = l5m.ConditionEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        condition_dim=16
    )
    decoder = l5m.ConditionFlatDecoder(
        condition_dim=16,
        latent_dim=16,
        output_shapes=OrderedDict(trajectories=(50, 3))
    )

    model = CVAE(
        q_encoder=q_encoder,
        c_encoder=c_encoder,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )

    inputs = OrderedDict(trajectories=torch.randn(10, 50, 3))
    conditions = OrderedDict(image=torch.randn(10, 3, 224, 224))

    outputs = model(inputs=inputs, condition_inputs=conditions)
    losses = model.compute_losses(outputs=outputs, targets=inputs)
    samples = model.sample(condition_inputs=conditions, n=10)
    print()


if __name__ == "__main__":
    main()