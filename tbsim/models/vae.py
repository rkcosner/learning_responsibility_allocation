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
        self._create_networks()

    def _create_networks(self):
        """Create networks for the prior module"""
        pass

    def forward(self, inputs=None):
        """Get prior parameters."""
        raise NotImplementedError

    def sample(self, n, inputs=None):
        """Take samples from the prior distribution."""
        raise NotImplementedError

    @staticmethod
    def sample_posterior(posterior_params, n: int):
        """Take samples using posterior distribution (q) parameters using reparameterization trick"""
        raise NotImplementedError

    def kl_loss(self, posterior_params, inputs=None):
        """Compute kl loss between the prior and the posteriors """
        raise NotImplementedError


class FixedGaussianPrior(Prior):
    """An unassuming unit Gaussian Prior"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(FixedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = {
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False)
        }

    def forward(self, inputs=None):
        return TensorUtils.clone(self._params)

    def sample(self, n, inputs=None):
        batch_size = 1
        if inputs is not None:
            assert isinstance(inputs, torch.Tensor)
            batch_size = inputs.shape[0]
        bs = batch_size * n
        mu = TensorUtils.unsqueeze_expand_at(self._params["mu"], size=bs, dim=0)
        logvar = TensorUtils.unsqueeze_expand_at(self._params["logvar"], size=bs, dim=0)
        samples = reparameterize(mu=mu, logvar=logvar)
        samples = TensorUtils.reshape_dimensions(samples, 0, 1, target_dims=(batch_size, n))
        return samples

    @staticmethod
    def sample_posterior(posterior_params, n: int):
        """Take samples using posterior distribution (q) parameters using reparameterization trick"""
        mu = posterior_params["mu"]
        batch_size = mu.shape[0]
        mu = TensorUtils.repeat_by_expand_at(mu, repeats=n, dim=0)
        logvar = TensorUtils.repeat_by_expand_at(posterior_params["logvar"], repeats=n, dim=0)
        samples = reparameterize(mu, logvar)
        samples = TensorUtils.reshape_dimensions(samples, begin_axis=0, end_axis=1, target_dims=(batch_size, n))
        return samples

    def kl_loss(self, posterior_params, inputs=None):
        assert posterior_params["mu"].shape[1] == self._latent_dim
        assert posterior_params["logvar"].shape[1] == self._latent_dim
        return KLD_0_1_loss(
            mu=posterior_params["mu"],
            logvar=posterior_params["logvar"]
        )


class LearnedGaussianPrior(FixedGaussianPrior):
    """A Gaussian prior with learnable parameters"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(LearnedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = {
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True)
        }

    def kl_loss(self, posterior_params, inputs=None):
        """Compute KL Divergence between two Gaussian distributions"""
        batch_size = posterior_params["mu"].shape[0]
        mu = TensorUtils.unsqueeze_expand_at(self._params["mu"], size=batch_size, dim=0)
        logvar = TensorUtils.unsqueeze_expand_at(self._params["logvar"], size=batch_size, dim=0)
        assert posterior_params["mu"].shape[1] == self._input_dim
        assert posterior_params["logvar"].shape[1] == self._input_dim
        return KLD_gaussian_loss(
            mu_1=posterior_params["mu"],
            logvar_1=posterior_params["logvar"],
            mu_2=mu,
            logvar_2=logvar
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
        Draw data samples (x') given a condition inputs (x_c) and the VAE prior
        Args:
            condition_inputs (dict): condition inputs - a dictionary of named tensors (x_c)
            n (int): number of samples to draw

        Returns:
            dictionary of batched samples (x')
        """
        c0 = self.c_encoder(condition_inputs)  # [B, ...]
        z = self.prior.sample(n=n, inputs=c0)  # z of shape [B (from c0), N, ...]
        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c0_samples = TensorUtils.repeat_by_expand_at(c0, repeats=n, dim=0)  # [B * N, ...]
        x_out = self.decoder(latents=z_samples, conditions=c0_samples)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c0.shape[0], n))
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
        z = self.prior.sample_posterior(q_params, n=1).squeeze(dim=1)
        c0 = self.c_encoder(condition_inputs)  # [B, ...]
        x_out = self.decoder(latents=z, conditions=c0)
        return {"x_recons": x_out, "q_params": q_params, "z": z}

    def compute_losses(self, inputs, condition_inputs, targets, target_weights=None, kl_weight: float = 1.0):
        """
        Compute VAE losses

        Args:
            inputs (dict): encoder inputs (x)
            condition_inputs (dict): condition inputs - a dictionary of named tensors (x_c)
            targets (dict): reconstruction targets (x')
            target_weights (dict): (Optional) a dictionary of floats for weighing loss for reconstructing individual
                                    elements in the target, must be the same shape as the target.
            kl_weight (float): weighting factor for the KL divergence loss

        Returns:
            a dictionary of loss values
        """
        outputs = self.forward(inputs=inputs, condition_inputs=condition_inputs)
        recon_loss = 0
        if target_weights is None:
            target_weights = dict()
        for k, v in targets.items():
            w = target_weights[k] if k in target_weights else torch.ones_like(targets[k])
            element_loss = self.target_criterion(outputs["x_recons"][k], targets[k]) * w
            recon_loss += torch.mean(element_loss)  # TODO: sum up last dimension instead of averaging all?

        kld_loss = self.prior.kl_loss(outputs["q_params"], inputs=condition_inputs)

        total_loss = recon_loss + kl_weight * kld_loss

        return {"vae_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kld_loss * kl_weight}


def main():
    import tbsim.models.l5kit_models as l5m
    prior = FixedGaussianPrior(latent_dim=16)

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        num_input_channels=3,
        visual_feature_size=128
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
        output_shapes=OrderedDict(target_positions=(50, 2), target_yaws=(50, 1))
    )

    model = CVAE(
        q_encoder=q_encoder,
        c_encoder=c_encoder,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )

    inputs = OrderedDict(target_positions=torch.randn(10, 50, 2), target_yaws=torch.randn(10, 50, 1))
    conditions = OrderedDict(image=torch.randn(10, 3, 224, 224))

    losses = model.compute_losses(inputs=inputs, condition_inputs=conditions, targets=inputs)
    samples = model.sample(condition_inputs=conditions, n=10)
    print()

if __name__ == "__main__":
    main()