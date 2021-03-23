""" Post-process regularization steps such as gradient normalizing. """

from torch.nn.utils import clip_grad_norm_

from skorch.callbacks import Callback
from skorch.utils import _unscale_optimizer_grads


__all__ = ['GradientNormClipping']


class GradientNormClipping(Callback):
    """Clips gradient norm of a module's parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified
    in-place.

    See :func:`torch.nn.utils.clip_grad_norm_` for more information.

    Parameters
    ----------
    gradient_clip_value : float (default=None)
      If not None, clip the norm of all model parameter gradients to this
      value. The type of the norm is determined by the
      ``gradient_clip_norm_type`` parameter and defaults to L2.

    gradient_clip_norm_type : float (default=2)
      Norm to use when gradient clipping is active. The default is
      to use L2-norm. Can be 'inf' for infinity norm.

    """
    def __init__(
            self,
            gradient_clip_value=None,
            gradient_clip_norm_type=2,
    ):
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_norm_type = gradient_clip_norm_type

    def on_grad_computed(self, net, named_parameters, **kwargs):
        """TODO"""
        if self.gradient_clip_value is None:
            return

        if net.amp_enabled:
            _unscale_optimizer_grads(net.grad_scaler_, net.optimizer_)

        clip_grad_norm_(
            (p for _, p in named_parameters),
            max_norm=self.gradient_clip_value,
            norm_type=self.gradient_clip_norm_type,
        )
