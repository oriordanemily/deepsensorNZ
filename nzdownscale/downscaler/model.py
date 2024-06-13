from deepsensor.model.convnp import ConvNP
import torch
from torch import nn
from typing import Tuple, Optional, Literal
from neuralprocesses.util import register_model
from neuralprocesses.architectures.util import construct_likelihood, parse_transform
from neuralprocesses.architectures.convgnp import (
    _convgnp_init_dims, 
    _convgnp_resolve_architecture, 
    _convgnp_assert_form_contexts, 
    _convgnp_construct_encoder_setconvs, 
    _convgnp_optional_division_by_density,
    _convgnp_construct_decoder_setconv
)
import neuralprocesses as nps  # This fixes inspection below.

import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import lab as B
from plum import convert

from neuralprocesses import _dispatch
from neuralprocesses.datadims import data_dims
from neuralprocesses.util import compress_batch_dimensions, register_module, with_first_last

# from deepsensor.deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.defaults import (
    compute_greatest_data_density,
    gen_encoder_scales,
    gen_decoder_scale,
)

class ConvNP_dropout(ConvNP):
    # def __init__(self, *args, **kwargs):

    # @dispatch
    # def __init__(self, *args, **kwargs):
    #     super(ConvNP_dropout, self).__init__(*args, **kwargs)
    #     """
    #     Generate a new model using ``construct_neural_process`` with default or
    #     specified parameters.

    #     This method does not take a ``TaskLoader`` or ``DataProcessor`` object,
    #     so the model will not auto-unnormalise predictions at inference time.
    #     """
    #     super().__init__()

    #     self.model, self.config = construct_neural_process(*args, **kwargs)

    # @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        *args,
        verbose: bool = True,
        **kwargs,
    ):
        super(ConvNP_dropout, self).__init__(*args, **kwargs)

        """
        Instantiate model from TaskLoader, using data to infer model parameters
        (unless overridden).

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                DataProcessor object. Used for unnormalising model predictions in
                ``.predict`` method.
            task_loader (:class:`~.data.loader.TaskLoader`):
                TaskLoader object. Used for inferring sensible defaults for hyperparameters
                that are not set by the user.
            verbose (bool, optional):
                Whether to print inferred model parameters, by default True.
        """
        super().__init__(data_processor, task_loader)

        if "dim_yc" not in kwargs:
            dim_yc = task_loader.context_dims
            if verbose:
                print(f"dim_yc inferred from TaskLoader: {dim_yc}")
            kwargs["dim_yc"] = dim_yc
        if "dim_yt" not in kwargs:
            dim_yt = sum(task_loader.target_dims)  # Must be an int
            if verbose:
                print(f"dim_yt inferred from TaskLoader: {dim_yt}")
            kwargs["dim_yt"] = dim_yt
        if "dim_aux_t" not in kwargs:
            dim_aux_t = task_loader.aux_at_target_dims
            if verbose:
                print(f"dim_aux_t inferred from TaskLoader: {dim_aux_t}")
            kwargs["dim_aux_t"] = dim_aux_t
        if "aux_t_mlp_layers" not in kwargs and kwargs["dim_aux_t"] > 0:
            kwargs["aux_t_mlp_layers"] = (64,) * 3
            if verbose:
                print(f"Setting aux_t_mlp_layers: {kwargs['aux_t_mlp_layers']}")
        if "internal_density" not in kwargs:
            internal_density = compute_greatest_data_density(task_loader)
            if verbose:
                print(f"internal_density inferred from TaskLoader: {internal_density}")
            kwargs["internal_density"] = internal_density
        if "encoder_scales" not in kwargs:
            encoder_scales = gen_encoder_scales(kwargs["internal_density"], task_loader)
            if verbose:
                print(f"encoder_scales inferred from TaskLoader: {encoder_scales}")
            kwargs["encoder_scales"] = encoder_scales
        if "decoder_scale" not in kwargs:
            decoder_scale = gen_decoder_scale(kwargs["internal_density"])
            if verbose:
                print(f"decoder_scale inferred from TaskLoader: {decoder_scale}")
            kwargs["decoder_scale"] = decoder_scale

        self.model, self.config = construct_neural_process(*args, **kwargs)
        self._set_num_mixture_components()

    # @dispatch
    # def __init__(
    #     self,
    #     data_processor: DataProcessor,
    #     task_loader: TaskLoader,
    #     neural_process: Union[TFModel, TorchModel],
    # ):
    #     super(ConvNP_dropout, self).__init__(*args, **kwargs)

    #     """
    #     Instantiate with a pre-defined neural process model.

    #     Args:
    #         data_processor (:class:`~.data.processor.DataProcessor`):
    #             DataProcessor object. Used for unnormalising model predictions in
    #             ``.predict`` method.
    #         task_loader (:class:`~.data.loader.TaskLoader`):
    #             TaskLoader object. Used for inferring sensible defaults for hyperparameters
    #             that are not set by the user.
    #         neural_process (TFModel | TorchModel):
    #             Pre-defined neural process PyTorch/TensorFlow model object.
    #     """
    #     super().__init__(data_processor, task_loader)

    #     self.model = neural_process
    #     self.config = None

    # @dispatch
    # def __init__(self, model_ID: str):
    #     super(ConvNP_dropout, self).__init__(*args, **kwargs)

    #     """Instantiate a model from a folder containing model weights and config."""
    #     super().__init__()

    #     self.load(model_ID)
        # self._set_num_mixture_components()

    # @dispatch
    # def __init__(
    #     self,
    #     data_processor: DataProcessor,
    #     task_loader: TaskLoader,
    #     model_ID: str,
    # ):
    #     super(ConvNP_dropout, self).__init__(*args, **kwargs)

    #     """Instantiate a model from a folder containing model weights and config.

    #     Args:
    #         data_processor (:class:`~.data.processor.DataProcessor`):
    #             dataprocessor object. used for unnormalising model predictions in
    #             ``.predict`` method.
    #         task_loader (:class:`~.data.loader.TaskLoader`):
    #             taskloader object. used for inferring sensible defaults for hyperparameters
    #             that are not set by the user.
    #         model_ID (str):
    #             folder to load the model config and weights from.
    #     """
    #     super().__init__(data_processor, task_loader)

    #     self.load(model_ID)
    #     self._set_num_mixture_components()

    def _set_num_mixture_components(self):
        """
        Set the number of mixture components for the model based on the likelihood.
        """
        if self.config["likelihood"] in ["spikes-beta"]:
            self.N_mixture_components = 3
        elif self.config["likelihood"] in ["bernoulli-gamma"]:
            self.N_mixture_components = 2
        else:
            self.N_mixture_components = 1

        # self.dropout = torch.nn.Dropout(p=0.5)
        
    # def forward(self, x, y, context):
    #     x = self.dropout(x)
    #     return super(ConvNP_dropout, self).forward(x, y, context)
    
def construct_neural_process(
    dim_x: int = 2,
    dim_yc: int = 1,
    dim_yt: int = 1,
    dim_aux_t: Optional[int] = None,
    dim_lv: int = 0,
    conv_arch: str = "unet",
    unet_channels: Tuple[int, ...] = (64, 64, 64, 64),
    unet_resize_convs: bool = True,
    unet_resize_conv_interp_method: Literal["bilinear"] = "bilinear",
    aux_t_mlp_layers: Optional[Tuple[int, ...]] = None,
    likelihood: Literal["cnp", "gnp", "cnp-spikes-beta"] = "cnp",
    unet_kernels: int = 5,
    internal_density: int = 100,
    encoder_scales: float = 1 / 100,
    encoder_scales_learnable: bool = False,
    decoder_scale: float = 1 / 100,
    decoder_scale_learnable: bool = False,
    num_basis_functions: int = 64,
    epsilon: float = 1e-2,
):
    """
    Construct a ``neuralprocesses`` ConvNP model.

    See: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/architectures/convgnp.py

    Docstring below modified from ``neuralprocesses``. If more kwargs are
    needed, they must be explicitly passed to ``neuralprocesses`` constructor
    (not currently safe to use `**kwargs` here).

    Args:
        dim_x (int, optional):
            Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional):
            Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional):
            Dimensionality of the outputs of the context set. You should set
            this if the dimensionality of the outputs of the context set is not
            equal to the dimensionality of the outputs of the target set. You
            should also set this if you want to use multiple context sets. In
            that case, set this equal to a tuple of integers indicating the
            respective output dimensionalities.
        dim_yt (int, optional):
            Dimensionality of the outputs of the target set. You should set
            this if the dimensionality of the outputs of the target set is not
            equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional):
            Dimensionality of target-specific auxiliary variables.
        internal_density (int, optional):
            Density of the ConvNP's internal grid (in terms of number of points
            per 1x1 unit square). Defaults to 100.
        likelihood (str, optional):
            Likelihood. Must be one of ``"cnp"`` (equivalently ``"het"``),
            ``"gnp"`` (equivalently ``"lowrank"``), or ``"cnp-spikes-beta"``
            (equivalently ``"spikes-beta"``). Defaults to ``"cnp"``.
        conv_arch (str, optional):
            Convolutional architecture to use. Must be one of
            ``"unet[-res][-sep]"`` or ``"conv[-res][-sep]"``. Defaults to
            ``"unet"``.
        unet_channels (tuple[int], optional):
            Channels of every layer of the UNet. Defaults to six layers each
            with 64 channels.
        unet_kernels (int or tuple[int], optional):
            Sizes of the kernels in the UNet. Defaults to 5.
        unet_resize_convs (bool, optional):
            Use resize convolutions rather than transposed convolutions in the
            UNet. Defaults to ``False``.
        unet_resize_conv_interp_method (str, optional):
            Interpolation method for the resize convolutions in the UNet. Can
            be set to ``"bilinear"``. Defaults to "bilinear".
        num_basis_functions (int, optional):
            Number of basis functions for the low-rank likelihood. Defaults to
            64.
        dim_lv (int, optional):
            Dimensionality of the latent variable. Setting to >0 constructs a
            latent neural process. Defaults to 0.
        encoder_scales (float or tuple[float], optional):
            Initial value for the length scales of the set convolutions for the
            context sets embeddings. Set to a tuple equal to the number of
            context sets to use different values for each set. Set to a single
            value to use the same value for all context sets. Defaults to
            ``1 / internal_density``.
        encoder_scales_learnable (bool, optional):
            Whether the encoder SetConv length scale(s) are learnable.
            Defaults to ``False``.
        decoder_scale (float, optional):
            Initial value for the length scale of the set convolution in the
            decoder. Defaults to ``1 / internal_density``.
        decoder_scale_learnable (bool, optional):
            Whether the decoder SetConv length scale(s) are learnable. Defaults
            to ``False``.
        aux_t_mlp_layers (tuple[int], optional):
            Widths of the layers of the MLP for the target-specific auxiliary
            variable. Defaults to three layers of width 128.
        epsilon (float, optional):
            Epsilon added by the set convolutions before dividing by the
            density channel. Defaults to ``1e-2``.

    Returns:
        :class:`.model.Model`:
            ConvNP model.

    Raises:
        NotImplementedError
            If specified backend has no default dtype.
    """
    if likelihood == "cnp":
        likelihood = "het"
    elif likelihood == "gnp":
        likelihood = "lowrank"
    elif likelihood == "cnp-spikes-beta":
        likelihood = "spikes-beta"
    elif likelihood == "cnp-bernoulli-gamma":
        likelihood = "bernoulli-gamma"

    # Log the call signature for `construct_convgnp`
    config = dict(locals())

    # if backend.str == "torch":
    import torch

    dtype = torch.float32
    # elif backend.str == "tf":
    #     import tensorflow as tf

    #     dtype = tf.float32
    # else:
    #     raise NotImplementedError(f"Backend {backend.str} has no default dtype.")

    neural_process = construct_convgnp(
        dim_x=dim_x,
        dim_yc=dim_yc,
        dim_yt=dim_yt,
        dim_aux_t=dim_aux_t,
        dim_lv=dim_lv,
        likelihood=likelihood,
        conv_arch=conv_arch,
        unet_channels=tuple(unet_channels),
        unet_resize_convs=unet_resize_convs,
        unet_resize_conv_interp_method=unet_resize_conv_interp_method,
        aux_t_mlp_layers=aux_t_mlp_layers,
        unet_kernels=unet_kernels,
        # Use a stride of 1 for the first layer and 2 for all other layers
        unet_strides=(1, *(2,) * (len(unet_channels) - 1)),
        points_per_unit=internal_density,
        encoder_scales=encoder_scales,
        encoder_scales_learnable=encoder_scales_learnable,
        decoder_scale=decoder_scale,
        decoder_scale_learnable=decoder_scale_learnable,
        num_basis_functions=num_basis_functions,
        epsilon=epsilon,
        dtype=dtype,
    )

    return neural_process, config

@register_model
def construct_convgnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_aux_t=None,
    points_per_unit=64,
    margin=0.1,
    likelihood="lowrank",
    conv_arch="unet",
    unet_channels=(64,) * 6,
    unet_kernels=5,
    unet_strides=2,
    unet_activations=None,
    unet_resize_convs=False,
    unet_resize_conv_interp_method="nearest",
    conv_receptive_field=None,
    conv_layers=6,
    conv_channels=64,
    num_basis_functions=64,
    dim_lv=0,
    lv_likelihood="het",
    encoder_scales=None,
    encoder_scales_learnable=True,
    decoder_scale=None,
    decoder_scale_learnable=True,
    aux_t_mlp_layers=(128,) * 3,
    divide_by_density=True,
    epsilon=1e-4,
    transform=None,
    dtype=None,
    nps=nps,
):
    """A Convolutional Gaussian Neural Process.

    Sets the attribute `receptive_field` to the receptive field of the model.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional): Dimensionality of target-specific auxiliary
            variables.
        points_per_unit (float, optional): Density of the internal discretisation.
            Defaults to 64.
        margin (float, optional): Margin of the internal discretisation. Defaults to
            0.1.
        likelihood (str, optional): Likelihood. Must be one of `"het"`, `"lowrank"`,
            `"spikes-beta"`, or `"bernoulli-gamma"`. Defaults to `"lowrank"`.
        conv_arch (str, optional): Convolutional architecture to use. Must be one of
            `"unet[-res][-sep]"` or `"conv[-res][-sep]"`. Defaults to `"unet"`.
        unet_channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels in the UNet.
            Defaults to 5.
        unet_strides (int or tuple[int], optional): Strides in the UNet. Defaults to 2.
        unet_activations (object or tuple[object], optional): Activation functions
            used by the UNet. If `None`, ReLUs are used.
        unet_resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions in the UNet. Defaults to `False`.
        unet_resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions in the UNet. Can be set to `"bilinear"`. Defaults
            to "nearest".
        conv_receptive_field (float, optional): Receptive field of the standard
            architecture. Must be specified if `conv_arch` is set to `"conv"`.
        conv_layers (int, optional): Layers of the standard architecture. Defaults to 8.
        conv_channels (int, optional): Channels of the standard architecture. Defaults
            to 64.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to `512`.
        dim_lv (int, optional): Dimensionality of the latent variable. Defaults to 0.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"` or `"lowrank"`. Defaults to `"het"`.
        encoder_scales (float or tuple[float], optional): Initial value for the length
            scales of the set convolutions for the context sets embeddings. Defaults
            to `1 / points_per_unit`.
        encoder_scales_learnable (bool, optional): Whether the encoder SetConv
            length scale(s) are learnable.
        decoder_scale (float, optional): Initial value for the length scale of the
            set convolution in the decoder. Defaults to `1 / points_per_unit`.
        decoder_scale_learnable (bool, optional): Whether the decoder SetConv
            length scale(s) are learnable.
        aux_t_mlp_layers (tuple[int], optional): Widths of the layers of the MLP
            for the target-specific auxiliary variable. Defaults to three layers of
            width 128.
        divide_by_density (bool, optional): Divide by the density channel. Defaults
            to `True`.
        epsilon (float, optional): Epsilon added by the set convolutions before
            dividing by the density channel. Defaults to `1e-4`.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: ConvGNP model.
    """
    dim_yc, dim_yt, conv_in_channels = _convgnp_init_dims(dim_yc, dim_yt, dim_y)

    # Construct likelihood of the encoder, which depends on whether we're using a
    # latent variable or not.
    if dim_lv > 0:
        lv_likelihood_in_channels, _, lv_likelihood = construct_likelihood(
            nps,
            spec=lv_likelihood,
            dim_y=dim_lv,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        encoder_likelihood = lv_likelihood
    else:
        encoder_likelihood = nps.DeterministicLikelihood()

    # Construct likelihood of the decoder.
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_yt,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )

    # Resolve the architecture.
    conv_out_channels = _convgnp_resolve_architecture(
        conv_arch,
        unet_channels,
        conv_channels,
        conv_receptive_field,
    )

    # If `dim_aux_t` is given, contruct an MLP which will use the auxiliary
    # information from the augmented inputs.
    if dim_aux_t:
        likelihood = nps.Augment(
            nps.Chain(
                MLP(
                    in_dim=conv_out_channels + dim_aux_t,
                    layers=aux_t_mlp_layers,
                    out_dim=likelihood_in_channels,
                    dtype=dtype,
                ),
                likelihood,
            )
        )
        linear_after_set_conv = lambda x: x  # See the `else` clause below.
    else:
        # There is no auxiliary MLP available, so the CNN will have to produce the
        # right number of channels. In this case, however, it may be more efficient
        # to produce the right number of channels _after_ the set conv.
        if conv_out_channels < likelihood_in_channels:
            # Perform an additional linear layer _after_ the set conv.
            linear_after_set_conv = Linear(
                in_channels=conv_out_channels,
                out_channels=likelihood_in_channels,
                dtype=dtype,
            )
        else:
            # Not necessary. Just let the CNN produce the right number of channels.
            conv_out_channels = likelihood_in_channels
            linear_after_set_conv = lambda x: x
        # Also assert that there is no augmentation given.
        likelihood = nps.Chain(nps.AssertNoAugmentation(), likelihood)

    # Construct the core CNN architectures for the encoder, which is only necessary
    # if we're using a latent variable, and for the decoder. First, we determine
    # how many channels these architectures should take in and produce.
    if dim_lv > 0:
        lv_in_channels = conv_in_channels
        lv_out_channels = lv_likelihood_in_channels
        in_channels = dim_lv
        out_channels = conv_out_channels  # These must be equal!
    else:
        in_channels = conv_in_channels
        out_channels = conv_out_channels  # These must be equal!
    if "unet" in conv_arch:
        if dim_lv > 0:
            lv_conv = UNet(
                dim=dim_x,
                in_channels=lv_in_channels,
                out_channels=lv_out_channels,
                channels=unet_channels,
                kernels=unet_kernels,
                strides=unet_strides,
                activations=unet_activations,
                resize_convs=unet_resize_convs,
                resize_conv_interp_method=unet_resize_conv_interp_method,
                separable="sep" in conv_arch,
                residual="res" in conv_arch,
                dtype=dtype,
            )
        else:
            lv_conv = lambda x: x

        conv = UNet(
            dim=dim_x,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=unet_channels,
            kernels=unet_kernels,
            strides=unet_strides,
            activations=unet_activations,
            resize_convs=unet_resize_convs,
            resize_conv_interp_method=unet_resize_conv_interp_method,
            separable="sep" in conv_arch,
            residual="res" in conv_arch,
            dtype=dtype,
        )
        receptive_field = conv.receptive_field / points_per_unit
    elif "conv" in conv_arch:
        if dim_lv > 0:
            lv_conv = ConvNet(
                dim=dim_x,
                in_channels=lv_in_channels,
                out_channels=lv_out_channels,
                channels=conv_channels,
                num_layers=conv_layers,
                points_per_unit=points_per_unit,
                receptive_field=conv_receptive_field,
                separable="sep" in conv_arch,
                residual="res" in conv_arch,
                dtype=dtype,
            )
        else:
            lv_conv = lambda x: x

        conv = ConvNet(
            dim=dim_x,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=conv_channels,
            num_layers=conv_layers,
            points_per_unit=points_per_unit,
            receptive_field=conv_receptive_field,
            separable="sep" in conv_arch,
            residual="res" in conv_arch,
            dtype=dtype,
        )
        receptive_field = conv_receptive_field
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # Construct the discretisation, taking into account that the input to the UNet
    # must play nice with the halving layers.
    disc = nps.Discretisation(
        points_per_unit=points_per_unit,
        multiple=2**conv.num_halving_layers,
        margin=margin,
        dim=dim_x,
    )

    # Construct model.
    model = nps.Model(
        nps.FunctionalCoder(
            disc,
            nps.Chain(
                _convgnp_assert_form_contexts(nps, dim_yc),
                nps.PrependDensityChannel(),
                _convgnp_construct_encoder_setconvs(
                    nps,
                    encoder_scales,
                    dim_yc,
                    disc,
                    dtype,
                    encoder_scales_learnable=encoder_scales_learnable,
                ),
                _convgnp_optional_division_by_density(nps, divide_by_density, epsilon),
                nps.Concatenate(),
                lv_conv,
                encoder_likelihood,
            ),
        ),
        nps.Chain(
            conv,
            nps.RepeatForAggregateInputs(
                nps.Chain(
                    _convgnp_construct_decoder_setconv(
                        nps,
                        decoder_scale,
                        disc,
                        dtype,
                        decoder_scale_learnable=decoder_scale_learnable,
                    ),
                    linear_after_set_conv,
                    selector,  # Select the right target output.
                )
            ),
            likelihood,
            parse_transform(nps, transform=transform),
        ),
    )

    # Set attribute `receptive_field`.
    model.receptive_field = receptive_field

    return model

__all__ = ["Linear", "MLP", "UNet", "ConvNet", "Conv", "ResidualBlock"]


@register_module
class Linear:
    """A linear layer over channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): Linear layer.
    """

    def __init__(self, in_channels, out_channels, dtype):
        self.nn = nps.torch.nn.Interface
        self.net = self.nn.Linear(in_channels, out_channels, dtype=dtype)


_nonlinearity_name_map = {
    "relu": "ReLU",
    "leakyrelu": "LeakyReLU",
}


@register_module
class MLP:
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (Callable or str, optional): Nonlinearity. Can also be specified
            as a string: `"ReLU"` or `"LeakyReLU"`. Defaults to ReLUs.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Union[Callable, str] = "ReLU",
        dtype=None,
    ):
        self.nn = nps.torch.nn.Interface
        # Check that one of the two specifications is given.
        layers_given = layers is not None
        num_layers_given = num_layers is not None and width is not None
        if not (layers_given or num_layers_given):
            raise ValueError(
                "Must specify either `layers` or `num_layers` and `width`."
            )
        # Make sure that `layers` is a tuple of various widths.
        if not layers_given and num_layers_given:
            layers = (width,) * num_layers

        # Resolve string-form `nonlinearity`.
        if isinstance(nonlinearity, str):
            try:
                resolved_name = _nonlinearity_name_map[nonlinearity.lower()]
                try:
                    nonlinearity = getattr(self.nn, resolved_name)()
                except:
                    nonlinearity = getattr(torch.nn, resolved_name)()
            except KeyError:
                raise ValueError(
                    f"Nonlinearity `{resolved_name}` invalid. "
                    f"Must be one of "
                    + ", ".join(f"`{k}`" for k in _nonlinearity_name_map.keys())
                    + "."
                )

        # Build layers.
        if len(layers) == 0:
            self.net = self.nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [self.nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(self.nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(self.nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = self.nn.Sequential(*net)

    def __call__(self, x):
        x = B.transpose(x)
        x, uncompress = compress_batch_dimensions(x, 2)
        x = self.net(x)
        x = uncompress(x)
        x = B.transpose(x)
        return x


@_dispatch
def code(coder: Union[Linear, MLP], xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)

    # Construct permutation to switch the channel dimension and the last dimension.
    switch = list(range(B.rank(z)))
    switch[-d - 1], switch[-1] = switch[-1], switch[-d - 1]

    # Switch, compress, apply network, uncompress, and undo switch.
    z = B.transpose(z, perm=switch)
    z, uncompress = compress_batch_dimensions(z, 2)
    z = coder.net(z)
    z = uncompress(z)
    z = B.transpose(z, perm=switch)

    return xz, z


@register_module
class UNet:
    """UNet.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        kernels (int or tuple[int], optional): Sizes of the kernels. Defaults to `5`.
        strides (int or tuple[int], optional): Strides. Defaults to `2`.
        activations (object or tuple[object], optional): Activation functions.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make residual convolutional blocks. Defaults to
            `False`.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
        strides (tuple[int]): Strides.
        activations (tuple[function]): Activation functions.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_fields (list[float]): Receptive field for every intermediate value.
        receptive_field (float): Receptive field of the model.
        before_turn_layers (list[module]): Layers before the U-turn.
        after_turn_layers (list[module]): Layers after the U-turn
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        kernels: Union[int, Tuple[Union[int, Tuple[int, ...]], ...]] = 5,
        strides: Union[int, Tuple[int, ...]] = 2,
        activations: Union[None, object, Tuple[object, ...]] = None,
        separable: bool = False,
        residual: bool = False,
        resize_convs: bool = False,
        resize_conv_interp_method: str = "nearest",
        dtype=None,
        dropout_rate=0.5
    ):
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.nn = nps.torch.nn.Interface 
        # If `kernel` is an integer, repeat it for every layer.
        if not isinstance(kernels, (tuple, list)):
            kernels = (kernels,) * len(channels)
        elif len(kernels) != len(channels):
            raise ValueError(
                f"Length of `kernels` ({len(kernels)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.kernels = kernels

        # If `strides` is an integer, repeat it for every layer.
        # TODO: Change the default so that the first stride is 1.
        if not isinstance(strides, (tuple, list)):
            strides = (strides,) * len(channels)
        elif len(strides) != len(channels):
            raise ValueError(
                f"Length of `strides` ({len(strides)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.strides = strides

        # Default to ReLUs. Moreover, if `activations` is an activation function, repeat
        # it for every layer.
        activations = activations or self.nn.ReLU()
        if not isinstance(activations, (tuple, list)):
            activations = (activations,) * len(channels)
        elif len(activations) != len(channels):
            raise ValueError(
                f"Length of `activations` ({len(activations)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.activations = activations

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for stride, kernel in zip(self.strides, self.kernels):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if stride > 1:
                if after_conv % 2 == 0:
                    # If even, then subsample.
                    self.receptive_fields.append(after_conv // 2)
                else:
                    # If odd, then average pool.
                    self.receptive_fields.append((after_conv + 1) // 2)
            else:
                self.receptive_fields.append(after_conv)
        # Backward pass:
        for stride, kernel in zip(reversed(self.strides), reversed(self.kernels)):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            if stride > 1:
                after_interp = self.receptive_fields[-1] * 2 - 1
                self.receptive_fields.append(after_interp + (kernel - 1))
            else:
                self.receptive_fields.append(self.receptive_fields[-1] + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        # If none of the fancy features are used, use the standard `self.nn.Conv` for
        # compatibility with trained models. For the same reason we also don't use the
        #   `activation` keyword.
        # TODO: In the future, use `self.nps.Conv` everywhere and use the `activation`
        #   keyword.
        if residual or separable or any(isinstance(k, tuple) for k in kernels):
            Conv = partial(
                self.nps.Conv,
                dim=dim,
                residual=residual,
                separable=separable,
            )
        else:

            def Conv(*, stride=1, transposed=False, **kw_args):
                if transposed and stride > 1:
                    kw_args["output_padding"] = stride // 2
                return self.nn.Conv(
                    dim=dim,
                    stride=stride,
                    transposed=transposed,
                    **kw_args,
                )

        def construct_before_turn_layer(i):
            # Determine the configuration of the layer.
            ci = ((in_channels,) + tuple(channels))[i]
            co = channels[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    dtype=dtype,
                )
            else:
                # This is a downsampling layer.
                if self.receptive_fields[i] % 2 == 1:
                    # Perform average pooling if the previous receptive field is odd.
                    return self.nn.Sequential(
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            dtype=dtype,
                        ),
                        self.nn.AvgPool(
                            dim=dim,
                            kernel=s,
                            stride=s,
                            dtype=dtype
                        ),
                    )
                else:
                    # Perform subsampling if the previous receptive field is even.
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        dtype=dtype,
                    )

        def construct_after_turn_layer(i):
            # Determine the configuration of the layer.
            if i == len(channels) - 1:
                # No skip connection yet.
                ci = channels[i]
            else:
                # Add the skip connection.
                ci = 2 * channels[i]
            co = ((channels[0],) + tuple(channels))[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    dtype=dtype,
                )
            else:
                # This is an upsampling layer.
                if resize_convs:
                    return self.nn.Sequential(
                        torch.nn.Upsample(
                            # dim=dim,
                            scale_factor=s,
                            mode=resize_conv_interp_method,
                        ),
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            dtype=dtype,
                        ),
                    )
                else:
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        transposed=True,
                        dtype=dtype,
                    )

        self.before_turn_layers = self.nn.ModuleList(
            [construct_before_turn_layer(i) for i in range(len(channels))]
        )
        self.after_turn_layers = self.nn.ModuleList(
            [construct_after_turn_layer(i) for i in range(len(channels))]
        )
        self.final_linear = self.nn.Conv(
            dim=dim,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
            dtype=dtype,
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        hs = [self.activations[0](self.before_turn_layers[0](x))]
        for layer, activation in zip(self.before_turn_layers[1:],
                                     self.activations[1:]):
            h = activation(layer(hs[-1]))
            h = torch.nn.Dropout(self.dropout_rate)(h)
            hs.append(h)

        # Now make the turn!
        h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer, activation in zip(reversed(hs[:-1]), 
                                             reversed(self.after_turn_layers[:-1]), 
                                             reversed(self.activations[:-1])):
            h_prev_dropped = torch.nn.Dropout(self.dropout_rate)(h_prev)
            h = activation(layer(B.concat(h_prev_dropped, h, axis=1)))
            h = torch.nn.Dropout(self.dropout_rate)(h)

        return uncompress(self.final_linear(h))

    # def __call__(self, x):
    #     x, uncompress = compress_batch_dimensions(x, self.dim + 1)

    #     hs = [self.activations[0](self.before_turn_layers[0](x))]
    #     for layer, activation in zip(
    #         self.before_turn_layers[1:],
    #         self.activations[1:],
    #     ):
    #         hs.append(activation(layer(hs[-1])))
    #         hs.append(torch.nn.Dropout(self.dropout_rate)(activation(layer(hs[-1]))))

    #     # Now make the turn!

    #     h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
    #     for h_prev, layer, activation in zip(
    #         reversed(hs[:-1]),
    #         reversed(self.after_turn_layers[:-1]),
    #         reversed(self.activations[:-1]),
    #     ):
    #         h = activation(layer(B.concat(h_prev, h, axis=1)))
    #         h = torch.nn.Dropout(torch.dropout_rate)(activation(layer(B.concat(h_prev, h, axis=1))))

    #     return uncompress(self.final_linear(h))


@register_module
class ConvNet:
    """A regular convolutional neural network.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of channels at every intermediate layer.
        num_layers (int): Number of layers.
        points_per_unit (float, optional): Density of the discretisation corresponding
            to the inputs.
        receptive_field (float, optional): Desired receptive field.
        kernel (int, optional): Kernel size. If set, then this overrides the computation
            done by `points_per_unit` and `receptive_field`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults
            to `True`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_field (float): Receptive field.
        conv_net (module): The architecture.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        kernel: Optional[int] = None,
        points_per_unit: Optional[float] = 1,
        receptive_field: Optional[float] = None,
        separable: bool = True,
        residual: bool = False,
        dtype=None,
    ):
        self.dim = dim
        self.nn = nps.torch.nn.Interface

        if kernel is None:
            # Compute kernel size.
            receptive_points = receptive_field * points_per_unit
            kernel = math.ceil(1 + (receptive_points - 1) / num_layers)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel  # Make kernel size odd.
            self.kernel = kernel  # Store it for reference.
        else:
            # Compute the receptive field size.
            receptive_points = kernel + num_layers * (kernel - 1)
            receptive_field = receptive_points / points_per_unit
            self.kernel = kernel

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0
        self.receptive_field = receptive_field

        # Construct basic building blocks.
        activation = self.nn.ReLU()

        self.conv_net = self.nn.Sequential(
            *(
                self.nps.Conv(
                    dim=dim,
                    in_channels=in_channels if first else channels,
                    out_channels=out_channels if last else channels,
                    kernel=kernel,
                    activation=None if first else activation,
                    separable=separable,
                    residual=residual,
                    dtype=dtype,
                )
                for first, last, _ in with_first_last(range(num_layers))
            )
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.conv_net(x))


@register_module
class Conv:
    """A flexible standard convolutional block.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int or tuple[int]): Kernel size(s). If it is a `tuple`, layers with
            those kernel sizes will be put in sequence.
        stride (int, optional): Stride.
        transposed (bool, optional): Transposed convolution. Defaults to `False`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make a residual block. Defaults to `False`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        net (object): Network.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, ...]],
        stride: int = 1,
        transposed: bool = False,
        activation=None,
        separable: bool = False,
        residual: bool = False,
        dtype=None,
    ):
        self.dim = dim

        if residual:
            self.net = self._init_residual(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                transposed=transposed,
                activation=activation,
                separable=separable,
                dtype=dtype,
            )
        else:
            if separable:
                self.net = self._init_separable_conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    dtype=dtype,
                )
            else:
                self.net = self._init_conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=1,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    dtype=dtype,
                )

    def _init_conv(
        self,
        dim,
        in_channels,
        out_channels,
        groups,
        kernel,
        stride,
        transposed,
        activation,
        dtype,
    ):
        intermediate_channels = min(in_channels, out_channels)
        self.nn = nps.torch.nn.Interface
        # Determine the output padding.
        if transposed and stride > 1:
            if stride % 2 == 0:
                output_padding = {"output_padding": stride // 2}
            else:
                raise RuntimeError(
                    "Can only set the output padding correctly for `stride`s "
                    "which are a multiple of two."
                )
        else:
            output_padding = {}

        # Prepend the activation, if one is given.
        if activation:
            net = [activation]
        else:
            net = []

        # If `kernel` is a `tuple`, concatenate so many layers.
        net.extend(
            [
                self.nn.Conv(
                    dim=dim,
                    in_channels=in_channels if first else intermediate_channels,
                    out_channels=out_channels if last else intermediate_channels,
                    groups=groups,
                    kernel=k,
                    stride=stride if last else 1,
                    transposed=transposed if last else 1,
                    **(output_padding if last else {}),
                    dtype=dtype,
                )
                for first, last, k in with_first_last(convert(kernel, tuple))
            ]
        )

        return self.nn.Sequential(*net)

    def _init_separable_conv(
        self,
        dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        transposed,
        activation,
        dtype,
    ):
        return self.nn.Sequential(
            self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                kernel=kernel,
                stride=stride,
                transposed=transposed,
                activation=activation,
                dtype=dtype,
            ),
            self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                groups=1,
                kernel=1,
                stride=1,
                transposed=False,
                activation=None,
                dtype=dtype,
            ),
        )

    def _init_residual(
        self,
        dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        transposed,
        activation,
        separable,
        dtype,
    ):
        intermediate_channels = min(in_channels, out_channels)
        if in_channels == intermediate_channels and stride == 1:
            # The input can be directly passed to the output.
            input_transform = lambda x: x
        else:
            # The input cannot be directly passed to the output, so we use an additional
            # linear layer.
            input_transform = self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=intermediate_channels,
                groups=1,
                kernel=1,
                stride=stride,
                transposed=transposed,
                activation=None,
                dtype=dtype,
            )
        return self.nps.ResidualBlock(
            input_transform,
            self.nn.Sequential(
                self.nps.Conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    separable=separable,
                    residual=False,
                    dtype=dtype,
                ),
                self.nn.ReLU(),
                self._init_conv(
                    dim=dim,
                    in_channels=intermediate_channels,
                    out_channels=intermediate_channels,
                    groups=1,
                    kernel=1,
                    stride=1,
                    transposed=False,
                    # TODO: Make this activation configurable.
                    activation=self.nn.ReLU(),
                    dtype=dtype,
                ),
            ),
            self._init_conv(
                dim=dim,
                in_channels=intermediate_channels,
                out_channels=out_channels,
                groups=1,
                kernel=1,
                stride=1,
                transposed=False,
                activation=None,
                dtype=dtype,
            ),
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.net(x))


@register_module
class ResidualBlock:
    """Block of a residual network.

    Args:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.

    Attributes:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.
    """

    def __init__(self, layer1, layer2, layer_post):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer_post = layer_post

    def __call__(self, x):
        return self.layer_post(self.layer1(x) + self.layer2(x))