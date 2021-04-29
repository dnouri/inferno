"""Custom scoring functions"""

import numpy as np

from skorch.dataset import unpack_data


def loss_scoring(net, X, y=None, sample_weight=None):
    """Calculate score using the criterion of the net

    Use the exact same logic as during model training to calculate the score.

    This function can be used to implement the ``score`` method for a
    :class:`.NeuralNet` through sub-classing. This is useful, for example, when
    combining skorch models with sklearn objects that rely on the model's
    ``score`` method. For example:

    >>> class ScoredNet(skorch.NeuralNetClassifier):
    ...     def score(self, X, y=None):
    ...         return loss_scoring(self, X, y)

    Parameters
    ----------
    net : skorch.NeuralNet
        A fitted Skorch :class:`.NeuralNet` object.

    X : input data, compatible with skorch.dataset.Dataset
        By default, you should be able to pass:

          * numpy arrays
          * torch tensors
          * pandas DataFrame or Series
          * scipy sparse CSR matrices
          * a dictionary of the former three
          * a list/tuple of the former three
          * a Dataset

        If this doesn't work with your data, you have to pass a
        ``Dataset`` that can deal with the data.

    y : target data, compatible with skorch.dataset.Dataset
        The same data types as for ``X`` are supported. If your X is a Dataset
        that contains the target, ``y`` may be set to None.

    sample_weight : array-like of shape (n_samples,)
        Sample weights.

    Returns
    -------
    loss_value : float32 or np.ndarray
        Return type depends on ``net.criterion_.reduction``, and will be a float
        if reduction is ``'sum'`` or ``'mean'``. If reduction is ``'none'`` then
        this function returns a ``np.ndarray`` object.

    """
    if sample_weight is not None:
        raise NotImplementedError(
            "sample_weight for loss_scoring is not yet supported."
        )

    net.check_is_fitted()

    dataset = net.get_dataset(X, y)
    iterator = net.get_iterator(dataset, training=False)
    history = {"loss": [], "batch_size": []}
    reduction = net.criterion_.reduction
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(
            "Expected one of 'mean', 'sum' or 'none' "
            "for reduction but got {reduction}.".format(reduction=reduction)
        )

    for batch in iterator:
        yp = net.evaluation_step(batch, training=False)
        yi = unpack_data(batch)[1]
        loss = net.get_loss(yp, yi)
        if reduction == "none":
            loss_value = loss.detach().cpu().numpy()
        else:
            loss_value = loss.item()
        history["loss"].append(loss_value)
        history["batch_size"].append(yi.size(0))

    if reduction == "none":
        return np.concatenate(history["loss"], 0)
    if reduction == "sum":
        return np.sum(history["loss"])
    return np.average(history["loss"], weights=history["batch_size"])
