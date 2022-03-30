"""Variants of Conditional Variational Autoencoder (C-VAE)"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tbsim.utils.loss_utils import KLD_0_1_loss, KLD_gaussian_loss, KLD_discrete
from tbsim.utils.torch_utils import reparameterize
import tbsim.utils.tensor_utils as TensorUtils
from torch.distributions import Categorical
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

    @staticmethod
    def get_mean(prior_params):
        """
        Extract the "mean" of a prior distribution (not supported by all priors)

        Args:
            prior_params (torch.Tensor): a batch of prior parameters

        Returns:
            mean (torch.Tensor): the "mean" of the distribution
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

    @property
    def latent_dim(self):
        """
        Shape of the latent code

        Returns:
            latent_dim (int)
        """
        return self._latent_dim


class FixedGaussianPrior(Prior):
    """An unassuming unit Gaussian Prior"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(FixedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = nn.ParameterDict({
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False)
        })

    @staticmethod
    def get_mean(prior_params):
        return prior_params["mu"]

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
            q_net: nn.Module,
            c_net: nn.Module,
            decoder: nn.Module,
            prior: Prior
    ):
        """
        A basic Conditional Variational Autoencoder Network (C-VAE)

        Args:
            q_net (nn.Module): a model that encodes data (x) and condition inputs (x_c) to posterior (q) parameters
            c_net (nn.Module): a model that encodes condition inputs (x_c) into condition feature (c)
            decoder (nn.Module): a model that decodes latent (z) and condition feature (c) to data (x')
            prior (nn.Module): a model containing information about distribution prior (kl-loss, prior params, etc.)
        """
        super(CVAE, self).__init__()
        self.q_net = q_net
        self.c_net = c_net
        self.decoder = decoder
        self.prior = prior

    def sample(self, condition_inputs, n: int, condition_feature=None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            n (int): number of samples to draw
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]
        z = self.prior.sample(n=n, inputs=c)  # z of shape [B (from c), N, ...]
        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * N, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c.shape[0], n))
        return x_out

    def predict(self, condition_inputs, condition_feature=None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]

        prior_params = self.prior(c)  # [B, ...]
        mu = self.prior.get_mean(prior_params)  # [B, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=mu, condition_features=c, **decoder_kwargs)
        return x_out

    def forward(self, inputs, condition_inputs, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """
        c = self.c_net(condition_inputs)  # [B, ...]
        q_params = self.q_net(inputs=inputs, condition_features=c)
        z = self.prior.sample_with_parameters(q_params, n=1).squeeze(dim=1)
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z, condition_features=c, **decoder_kwargs)
        return {"x_recons": x_out, "q_params": q_params, "z": z, "c": c}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        return self.prior.kl_loss(outputs["q_params"], inputs=outputs["c"])

class DiscreteCVAE(nn.Module):
    def __init__(
            self,
            q_net: nn.Module,
            p_net: nn.Module,
            c_net: nn.Module,
            decoder: nn.Module,
            K: int,
            recon_loss_fun=None,
    ):
        """
        A basic Conditional Variational Autoencoder Network (C-VAE)

        Args:
            q_net (nn.Module): a model that encodes data (x) and condition inputs (x_c) to posterior (q) parameters
            p_net (nn.Module): a model that encodes condition feature (c) to latent (p) parameters
            c_net (nn.Module): a model that encodes condition inputs (x_c) into condition feature (c)
            decoder (nn.Module): a model that decodes latent (z) and condition feature (c) to data (x')
            K (int): cardinality of the discrete latent
            recon_loss: loss function handle for reconstruction loss
        """
        super(DiscreteCVAE, self).__init__()
        self.q_net = q_net
        self.p_net = p_net
        self.c_net = c_net
        self.decoder = decoder
        self.K = K
        if recon_loss_fun is None:
            self.recon_loss_fun = nn.MSELoss(reduction="none")
        else:
            self.recon_loss_fun = recon_loss_fun

    def sample(self, condition_inputs, n: int, condition_feature=None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            n (int): number of samples to draw
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        assert n<=self.K
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]
        logp = self.p_net(c)["logp"]
        # p = torch.exp(logp)
        # p = p/p.sum(dim=-1,keepdim=True)
        z = (-logp).argsort()[...,:n]
        z = F.one_hot(z,self.K)

        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * N, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c.shape[0], n))
        return x_out

    def predict(self, condition_inputs, condition_feature=None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]

        logp = self.p_net(c)["logp"]
        z = logp.argmax(dim=-1)
        
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=F.one_hot(z,self.K), condition_features=c, **decoder_kwargs)
        return x_out

    def forward(self, inputs, condition_inputs, n=None, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            n (int): number of samples, if not given, then n=self.K
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """
        if n is None:
            n = self.K
        c = self.c_net(condition_inputs)  # [B, ...]
        logq = self.q_net(inputs=inputs, condition_features=c)["logq"]
        q = torch.exp(logq)
        q = q/q.sum(dim=-1,keepdim=True)
        logp = self.p_net(c)["logp"]
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)
        z = (-logq).argsort()[...,:n]
        z = F.one_hot(z,self.K)
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        c_tiled = c.unsqueeze(1).repeat(1,n,1)
        x_out = self.decoder(latents=z.reshape(-1,self.K), condition_features=c_tiled.reshape(-1,c.shape[-1]), **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(z.shape[0],n))
        return {"x_recons": x_out, "q": q, "p": p, "z": z, "c": c}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        p = outputs["p"]
        q = outputs["q"]
        return (p*(torch.log(p)-torch.log(q))).sum(dim=-1)
    def compute_losses(self,outputs,targets,gamma=1):
        recon_loss = 0
        for k,v in outputs['x_recons'].items():
            if k in targets:
                if isinstance(self.recon_loss_fun,dict):
                    loss_v = self.recon_loss_fun[k](v,targets[k].unsqueeze(1))
                else:
                    loss_v = self.recon_loss_fun(v,targets[k].unsqueeze(1))
                sum_dim=tuple(range(2,loss_v.ndim))
                loss_v = loss_v.sum(dim=sum_dim)
                loss_v_detached = loss_v.detach()
                min_flag = (loss_v==loss_v.min(dim=1,keepdim=True)[0])
                nonmin_flag = torch.logical_not(min_flag)
                recon_loss +=(loss_v*min_flag*outputs["q"]).sum(dim=1)+(loss_v_detached*nonmin_flag*outputs["q"]).sum(dim=1)

        KL_loss = self.compute_kl_loss(outputs)
        return recon_loss + gamma*KL_loss







def main():
    import tbsim.models.base_models as l5m

    inputs = OrderedDict(trajectories=torch.randn(10, 50, 3))
    condition_inputs = OrderedDict(image=torch.randn(10, 3, 224, 224))

    condition_dim = 16
    latent_dim = 4

    prior = FixedGaussianPrior(latent_dim=4)

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        num_input_channels=3,
        feature_dim=128
    )

    q_encoder = l5m.PosteriorEncoder(
        condition_dim=condition_dim,
        trajectory_shape=(50, 3),
        output_shapes=OrderedDict(mu=(latent_dim,), logvar=(latent_dim,))
    )
    c_encoder = l5m.ConditionEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        condition_dim=condition_dim
    )
    decoder = l5m.ConditionDecoder(
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        output_shapes=OrderedDict(trajectories=(50, 3))
    )

    model = CVAE(
        q_net=q_encoder,
        c_net=c_encoder,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )


    outputs = model(inputs=inputs, condition_inputs=condition_inputs)
    losses = model.compute_losses(outputs=outputs, targets=inputs)
    samples = model.sample(condition_inputs=condition_inputs, n=10)
    print()

    traj_encoder = l5m.RNNTrajectoryEncoder(
        trajectory_dim=3,
        rnn_hidden_size=100
    )

    c_net = l5m.ConditionNet(
        condition_input_shapes=OrderedDict(
            map_feature=(map_encoder.output_shape()[-1],)
        ),
        condition_dim=condition_dim,
    )

    q_net = l5m.PosteriorNet(
        input_shapes=OrderedDict(
            traj_feature=(traj_encoder.output_shape()[-1],)
        ),
        condition_dim=condition_dim,
        param_shapes=prior.posterior_param_shapes,
    )

    lean_model = CVAE(
        q_net=q_net,
        c_net=c_net,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )

    map_feats = map_encoder(condition_inputs["image"])
    traj_feats = traj_encoder(inputs["trajectories"])
    input_feats = dict(traj_feature=traj_feats)
    condition_feats = dict(map_feature=map_feats)

    outputs = lean_model(inputs=input_feats, condition_inputs=condition_feats)
    losses = lean_model.compute_losses(outputs=outputs, targets=inputs)
    samples = lean_model.sample(condition_inputs=condition_feats, n=10)
    print()

def main_discrete():
    import tbsim.models.base_models as l5m

    inputs = OrderedDict(trajectories=torch.randn(10, 50, 3))
    condition_inputs = OrderedDict(image=torch.randn(10, 3, 224, 224))

    condition_dim = 16
    latent_dim = 20

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        feature_dim=128
    )

    q_encoder = l5m.PosteriorEncoder(
        condition_dim=condition_dim,
        trajectory_shape=(50, 3),
        output_shapes=OrderedDict(logq=(latent_dim,))
    )
    p_encoder = l5m.SplitMLP(
                input_dim=condition_dim,
                layer_dims=(128,128),
                output_shapes=OrderedDict(logp=(latent_dim,))
            )
    c_encoder = l5m.ConditionEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        condition_dim=condition_dim
    )
    decoder_MLP = l5m.SplitMLP(
        input_dim=condition_dim+latent_dim,
        output_shapes=OrderedDict(trajectories=(50, 3)),
        layer_dims=(128,128),
        output_activation=nn.ReLU,
    )
    decoder = l5m.ConditionDecoder(decoder_model=decoder_MLP)

    model = DiscreteCVAE(
        q_net=q_encoder,
        p_net=p_encoder,
        c_net=c_encoder,
        decoder=decoder,
        K=latent_dim,
    )


    outputs = model(inputs=inputs, condition_inputs=condition_inputs)
    losses = model.compute_losses(outputs=outputs, targets = inputs)
    samples = model.sample(condition_inputs=condition_inputs, n=10)
    KL_loss = model.compute_kl_loss(outputs)


    # traj_encoder = l5m.RNNTrajectoryEncoder(
    #     trajectory_dim=3,
    #     rnn_hidden_size=100
    # )

    # c_net = l5m.ConditionNet(
    #     condition_input_shapes=OrderedDict(
    #         map_feature=(map_encoder.output_shape()[-1],)
    #     ),
    #     condition_dim=condition_dim,
    # )

    # q_net = l5m.PosteriorNet(
    #     input_shapes=OrderedDict(
    #         traj_feature=(traj_encoder.output_shape()[-1],)
    #     ),
    #     condition_dim=condition_dim,
    #     param_shapes=prior.posterior_param_shapes,
    # )

    # lean_model = CVAE(
    #     q_net=q_net,
    #     c_net=c_net,
    #     decoder=decoder,
    #     prior=prior,
    #     target_criterion=nn.MSELoss(reduction="none")
    # )

    # map_feats = map_encoder(condition_inputs["image"])
    # traj_feats = traj_encoder(inputs["trajectories"])
    # input_feats = dict(traj_feature=traj_feats)
    # condition_feats = dict(map_feature=map_feats)

    # outputs = lean_model(inputs=input_feats, condition_inputs=condition_feats)
    # losses = lean_model.compute_losses(outputs=outputs, targets=inputs)
    # samples = lean_model.sample(condition_inputs=condition_feats, n=10)
    # print()

if __name__ == "__main__":
    main_discrete()