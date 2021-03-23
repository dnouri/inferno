"""Neural net classes."""

import fnmatch
from itertools import chain
from collections import OrderedDict
import tempfile
import warnings

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader

from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import PassthroughScoring
from skorch.dataset import Dataset
from skorch.dataset import CVSplit
from skorch.dataset import get_len
from skorch.dataset import unpack_data
from skorch.dataset import uses_placeholder_y
from skorch.exceptions import DeviceWarning
from skorch.history import History
from skorch.setter import optimizer_setter
from skorch.utils import _identity
from skorch.utils import _infer_predict_nonlinearity
from skorch.utils import FirstStepAccumulator
from skorch.utils import TeeGenerator
from skorch.utils import _check_f_arguments
from skorch.utils import check_is_fitted
from skorch.utils import duplicate_items
from skorch.utils import get_map_location
from skorch.utils import is_dataset
from skorch.utils import no_context
from skorch.utils import params_for
from skorch.utils import to_device
from skorch.utils import to_numpy
from skorch.utils import to_tensor


_PYTORCH_COMPONENTS = {'criterion', 'module', 'optimizer'}
"""Special names that mark pytorch components.

These special names are used to recognize whether an attribute that is
being set in the net should be added to prefixes_ and
cuda_dependent_attributes_

"""


# pylint: disable=too-many-instance-attributes
class NeuralNet:
    # pylint: disable=anomalous-backslash-in-string
    """NeuralNet base class.

    The base class covers more generic cases. Depending on your use
    case, you might want to use :class:`.NeuralNetClassifier` or
    :class:`.NeuralNetRegressor`.

    In addition to the parameters listed below, there are parameters
    with specific prefixes that are handled separately. To illustrate
    this, here is an example:

    >>> net = NeuralNet(
    ...    ...,
    ...    optimizer=torch.optimizer.SGD,
    ...    optimizer__momentum=0.95,
    ...)

    This way, when ``optimizer`` is initialized, :class:`.NeuralNet`
    will take care of setting the ``momentum`` parameter to 0.95.

    (Note that the double underscore notation in
    ``optimizer__momentum`` means that the parameter ``momentum``
    should be set on the object ``optimizer``. This is the same
    semantic as used by sklearn.)

    Furthermore, this allows to change those parameters later:

    ``net.set_params(optimizer__momentum=0.99)``

    This can be useful when you want to change certain parameters
    using a callback, when using the net in an sklearn grid search,
    etc.

    By default an :class:`.EpochTimer`, :class:`.BatchScoring` (for
    both training and validation datasets), and :class:`.PrintLog`
    callbacks are installed for the user's convenience.

    Parameters
    ----------
    module : torch module (class or instance)
      A PyTorch :class:`~torch.nn.Module`. In general, the
      uninstantiated class should be passed, although instantiated
      modules will also work.

    criterion : torch criterion (class)
      The uninitialized criterion (loss) used to optimize the
      module.

    optimizer : torch optim (class, default=torch.optim.SGD)
      The uninitialized optimizer (update rule) used to optimize the
      module

    lr : float (default=0.01)
      Learning rate passed to the optimizer. You may use ``lr`` instead
      of using ``optimizer__lr``, which would result in the same outcome.

    max_epochs : int (default=10)
      The number of epochs to train for each ``fit`` call. Note that you
      may keyboard-interrupt training at any time.

    batch_size : int (default=128)
      Mini-batch size. Use this instead of setting
      ``iterator_train__batch_size`` and ``iterator_test__batch_size``,
      which would result in the same outcome. If ``batch_size`` is -1,
      a single batch with all the data will be used during training
      and validation.

    iterator_train : torch DataLoader
      The default PyTorch :class:`~torch.utils.data.DataLoader` used for
      training data.

    iterator_valid : torch DataLoader
      The default PyTorch :class:`~torch.utils.data.DataLoader` used for
      validation and test data, i.e. during inference.

    dataset : torch Dataset (default=skorch.dataset.Dataset)
      The dataset is necessary for the incoming data to work with
      pytorch's ``DataLoader``. It has to implement the ``__len__`` and
      ``__getitem__`` methods. The provided dataset should be capable of
      dealing with a lot of data types out of the box, so only change
      this if your data is not supported. You should generally pass the
      uninitialized ``Dataset`` class and define additional arguments to
      X and y by prefixing them with ``dataset__``. It is also possible
      to pass an initialzed ``Dataset``, in which case no additional
      arguments may be passed.

    train_split : None or callable (default=skorch.dataset.CVSplit(5))
      If None, there is no train/validation split. Else, train_split
      should be a function or callable that is called with X and y
      data and should return the tuple ``dataset_train, dataset_valid``.
      The validation data may be None.

    callbacks : None, "disable", or list of Callback instances (default=None)
      Which callbacks to enable. There are three possible values:

      If ``callbacks=None``, only use default callbacks,
      those returned by ``get_default_callbacks``.

      If ``callbacks="disable"``, disable all callbacks, i.e. do not run
      any of the callbacks.

      If ``callbacks`` is a list of callbacks, use those callbacks in
      addition to the default callbacks. Each callback should be an
      instance of :class:`.Callback`.

      Callback names are inferred from the class
      name. Name conflicts are resolved by appending a count suffix
      starting with 1, e.g. ``EpochScoring_1``. Alternatively,
      a tuple ``(name, callback)`` can be passed, where ``name``
      should be unique. Callbacks may or may not be instantiated.
      The callback name can be used to set parameters on specific
      callbacks (e.g., for the callback with name ``'print_log'``, use
      ``net.set_params(callbacks__print_log__keys_ignored=['epoch',
      'train_loss'])``).

    predict_nonlinearity : callable, None, or 'auto' (default='auto')
      The nonlinearity to be applied to the prediction. When set to
      'auto', infers the correct nonlinearity based on the criterion
      (softmax for :class:`~torch.nn.CrossEntropyLoss` and sigmoid for
      :class:`~torch.nn.BCEWithLogitsLoss`). If it cannot be inferred
      or if the parameter is None, just use the identity
      function. Don't pass a lambda function if you want the net to be
      pickleable.

      In case a callable is passed, it should accept the output of the
      module (the first output if there is more than one), which is a
      PyTorch tensor, and return the transformed PyTorch tensor.

      This can be useful, e.g., when
      :func:`~skorch.NeuralNetClassifier.predict_proba`
      should return probabilities but a criterion is used that does
      not expect probabilities. In that case, the module can return
      whatever is required by the criterion and the
      ``predict_nonlinearity`` transforms this output into
      probabilities.

      The nonlinearity is applied only when calling
      :func:`~skorch.classifier.NeuralNetClassifier.predict` or
      :func:`~skorch.classifier.NeuralNetClassifier.predict_proba` but
      not anywhere else -- notably, the loss is unaffected by this
      nonlinearity.

    warm_start : bool (default=False)
      Whether each fit call should lead to a re-initialization of the
      module (cold start) or whether the module should be trained
      further (warm start).

    verbose : int (default=1)
      This parameter controls how much print output is generated by
      the net and its callbacks. By setting this value to 0, e.g. the
      summary scores at the end of each epoch are no longer printed.
      This can be useful when running a hyperparameter search. The
      summary scores are always logged in the history attribute,
      regardless of the verbose setting.

    device : str, torch.device (default='cpu')
      The compute device to be used. If set to 'cuda', data in torch
      tensors will be pushed to cuda tensors before being sent to the
      module. If set to None, then all compute devices will be left
      unmodified.

    Attributes
    ----------
    prefixes_ : list of str
      Contains the prefixes to special parameters. E.g., since there
      is the ``'module'`` prefix, it is possible to set parameters like
      so: ``NeuralNet(..., optimizer__momentum=0.95)``.

    cuda_dependent_attributes_ : list of str
      Contains a list of all attribute prefixes whose values depend on a
      CUDA device. If a ``NeuralNet`` trained with a CUDA-enabled device
      is unpickled on a machine without CUDA or with CUDA disabled, the
      listed attributes are mapped to CPU.  Expand this list if you
      want to add other cuda-dependent attributes.

    initialized_ : bool
      Whether the :class:`.NeuralNet` was initialized.

    module_ : torch module (instance)
      The instantiated module.

    criterion_ : torch criterion (instance)
      The instantiated criterion.

    callbacks_ : list of tuples
      The complete (i.e. default and other), initialized callbacks, in
      a tuple with unique names.

    amp_enabled TODO

    grad_scaler TODO

    """
    prefixes_ = ['module', 'iterator_train', 'iterator_valid', 'optimizer',
                 'criterion', 'callbacks', 'dataset', 'grad_scaler']

    cuda_dependent_attributes_ = ['module_', 'optimizer_', 'criterion_', 'grad_scaler_']

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            module,
            criterion,
            optimizer=torch.optim.SGD,
            lr=0.01,
            max_epochs=10,
            batch_size=128,
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            dataset=Dataset,
            train_split=CVSplit(5),
            callbacks=None,
            predict_nonlinearity='auto',
            warm_start=False,
            verbose=1,
            device='cpu',
            amp_enabled=False,
            grad_scaler=None,
            **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.iterator_train = iterator_train
        self.iterator_valid = iterator_valid
        self.dataset = dataset
        self.train_split = train_split
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.amp_enabled = amp_enabled
        self.grad_scaler = grad_scaler

        self._check_deprecated_params(**kwargs)
        history = kwargs.pop('history', None)
        initialized = kwargs.pop('initialized_', False)
        virtual_params = kwargs.pop('virtual_params_', dict())

        kwargs = self._check_kwargs(kwargs)
        vars(self).update(kwargs)

        self.history_ = history
        self.initialized_ = initialized
        self.virtual_params_ = virtual_params

    @property
    def history(self):
        return self.history_

    @history.setter
    def history(self, value):
        self.history_ = value

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('print_log', PrintLog()),
        ]

    def get_default_callbacks(self):
        return self._default_callbacks

    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in ``method_name`` with
        parameters specified in ``cb_kwargs``.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_epoch_begin
        * on_epoch_end
        * on_batch_begin
        * on_batch_end

        """
        getattr(self, method_name)(self, **cb_kwargs)
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)

    # pylint: disable=unused-argument
    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_train_end(self, net,
                     X=None, y=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))

    # pylint: disable=unused-argument
    def on_epoch_end(self, net,
                     dataset_train=None, dataset_valid=None, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_batch_begin(self, net,
                       Xi=None, yi=None, training=False, **kwargs):
        self.history.new_batch()

    def on_batch_end(self, net,
                     Xi=None, yi=None, training=False, **kwargs):
        pass

    def on_grad_computed(self, net, named_parameters,
                         Xi=None, yi=None,
                         training=False, **kwargs):
        pass

    def _yield_callbacks(self):
        """Yield all callbacks set on this instance including
        a set whether its name was set by the user.

        Handles these cases:
          * default and user callbacks
          * callbacks with and without name
          * initialized and uninitialized callbacks
          * puts PrintLog(s) last

        """
        print_logs = []
        for item in self.get_default_callbacks() + (self.callbacks or []):
            if isinstance(item, (tuple, list)):
                named_by_user = True
                name, cb = item
            else:
                named_by_user = False
                cb = item
                if isinstance(cb, type):  # uninitialized:
                    name = cb.__name__
                else:
                    name = cb.__class__.__name__
            if isinstance(cb, PrintLog) or (cb == PrintLog):
                print_logs.append((name, cb, named_by_user))
            else:
                yield name, cb, named_by_user
        yield from print_logs

    def _callbacks_grouped_by_name(self):
        """Group callbacks by name and collect names set by the user."""
        callbacks, names_set_by_user = OrderedDict(), set()
        for name, cb, named_by_user in self._yield_callbacks():
            if named_by_user:
                names_set_by_user.add(name)
            callbacks[name] = callbacks.get(name, []) + [cb]
        return callbacks, names_set_by_user

    def _uniquely_named_callbacks(self):
        """Make sure that the returned dict of named callbacks is unique
        w.r.t. to the callback name. User-defined names will not be
        renamed on conflict, instead an exception will be raised. The
        same goes for the event where renaming leads to a conflict.
        """
        grouped_cbs, names_set_by_user = self._callbacks_grouped_by_name()
        for name, cbs in grouped_cbs.items():
            if len(cbs) > 1 and name in names_set_by_user:
                raise ValueError("Found duplicate user-set callback name "
                                 "'{}'. Use unique names to correct this."
                                 .format(name))

            for i, cb in enumerate(cbs):
                if len(cbs) > 1:
                    unique_name = '{}_{}'.format(name, i+1)
                    if unique_name in grouped_cbs:
                        raise ValueError("Assigning new callback name failed "
                                         "since new name '{}' exists already."
                                         .format(unique_name))
                else:
                    unique_name = name
                yield unique_name, cb

    def initialize_callbacks(self):
        """Initializes all callbacks and save the result in the
        ``callbacks_`` attribute.

        Both ``default_callbacks`` and ``callbacks`` are used (in that
        order). Callbacks may either be initialized or not, and if
        they don't have a name, the name is inferred from the class
        name. The ``initialize`` method is called on all callbacks.

        The final result will be a list of tuples, where each tuple
        consists of a name and an initialized callback. If names are
        not unique, a ValueError is raised.

        """
        if self.callbacks == "disable":
            self.callbacks_ = []
            return self

        callbacks_ = []

        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        for name, cb in self._uniquely_named_callbacks():
            # check if callback itself is changed
            param_callback = getattr(self, 'callbacks__' + name, Dummy)
            if param_callback is not Dummy:  # callback itself was set
                cb = param_callback

            # below: check for callback params
            # don't set a parameter for non-existing callback
            params = self.get_params_for('callbacks__{}'.format(name))
            if (cb is None) and params:
                raise ValueError("Trying to set a parameter for callback {} "
                                 "which does not exist.".format(name))
            if cb is None:
                continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            cb.initialize()
            callbacks_.append((name, cb))

        self.callbacks_ = callbacks_

        return self

    def initialize_criterion(self):
        """Initializes the criterion."""
        criterion_params = self.get_params_for('criterion')
        self.criterion_ = self.criterion(**criterion_params)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_ = to_device(self.criterion_, self.device)
        return self

    def _format_reinit_msg(self, name, kwargs=None, triggered_directly=True):
        """Returns a message that informs about re-initializing a compoment.

        Sometimes, the module or optimizer need to be
        re-initialized. Not only should the user receive a message
        about this but also should they be informed about what
        parameters, if any, caused it.

        """
        msg = "Re-initializing {}".format(name)
        if triggered_directly and kwargs:
            msg += (" because the following parameters were re-set: {}."
                    .format(', '.join(sorted(kwargs))))
        else:
            msg += "."
        return msg

    def initialize_module(self):
        """Initializes the module.

        Note that if the module has learned parameters, those will be
        reset.

        """
        kwargs = self.get_params_for('module')
        module = self.module
        is_initialized = isinstance(module, torch.nn.Module)

        if kwargs or not is_initialized:
            if is_initialized:
                module = type(module)

            if (is_initialized or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("module", kwargs)
                print(msg)

            module = module(**kwargs)

        self.module_ = to_device(module, self.device)
        return self

    def _is_virtual_param(self, key):
        return any(fnmatch.fnmatch(key, pat) for pat in self.virtual_params_)

    def _virtual_setattr(self, param, val):
        setattr(self, param, val)

    def _register_virtual_param(self, param_patterns, fn=_virtual_setattr):
        if not isinstance(param_patterns, list):
            param_patterns = [param_patterns]
        for pattern in param_patterns:
            self.virtual_params_[pattern] = fn

    def _apply_virtual_params(self, virtual_kwargs):
        for pattern, fn in self.virtual_params_.items():
            for key, val in virtual_kwargs.items():
                if not fnmatch.fnmatch(key, pattern):
                    continue
                fn(self, key, val)

    def initialize_virtual_params(self):
        self.virtual_params_ = {}

    def initialize_optimizer(self, triggered_directly=True):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly : bool (default=True)
          Only relevant when optimizer is re-initialized.
          Initialization of the optimizer can be triggered directly
          (e.g. when lr was changed) or indirectly (e.g. when the
          module was re-initialized). If and only if the former
          happens, the user should receive a message informing them
          about the parameters that caused the re-initialization.

        """
        args, kwargs = self.get_params_for_optimizer(
            'optimizer', self.module_.named_parameters())

        if self.initialized_ and self.verbose:
            msg = self._format_reinit_msg(
                "optimizer", kwargs, triggered_directly=triggered_directly)
            print(msg)

        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr

        self.optimizer_ = self.optimizer(*args, **kwargs)

        self._register_virtual_param(
            ['optimizer__param_groups__*__*', 'optimizer__*', 'lr'],
            optimizer_setter,
        )

    def initialize_history(self):
        """Initializes the history."""
        self.history_ = History()

    def initialize_grad_scaler(self):
        if not self.amp_enabled:
            self.grad_scaler_ = None
            return self

        kwargs = self.get_params_for('grad_scaler')
        grad_scaler = self.grad_scaler if self.grad_scaler else torch.cuda.amp.GradScaler
        self.grad_scaler_ = grad_scaler(**kwargs)
        return self

    def initialize(self):
        """Initializes all components of the :class:`.NeuralNet` and
        returns self.

        """
        self.initialize_virtual_params()
        self.initialize_callbacks()
        self.initialize_criterion()
        self.initialize_module()
        self.initialize_optimizer()
        self.initialize_history()
        self.initialize_grad_scaler()

        self.initialized_ = True
        return self

    def check_data(self, X, y=None):
        pass

    def validation_step(self, Xi, yi, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.module_.eval()
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
            }

    def train_step_single(self, Xi, yi, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.module_.train()
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=Xi,
            y=yi
        )

        return {
            'loss': loss,
            'y_pred': y_pred,
            }

    def get_train_step_accumulator(self):
        """Return the train step accumulator.

        By default, the accumulator stores and retrieves the first
        value from the optimizer call. Most optimizers make only one
        call, so first value is at the same time the only value.

        In case of some optimizers, e.g. LBFGS,
        ``train_step_calc_gradient`` is called multiple times, as the
        loss function is evaluated multiple times per optimizer
        call. If you don't want to return the first value in that
        case, override this method to return your custom accumulator.

        """
        return FirstStepAccumulator()

    def train_step(self, Xi, yi, **fit_params):
        """Prepares a loss function callable and pass it to the optimizer,
        hence performing one optimization step.

        Loss function callable as required by some optimizers (and accepted by
        all of them):
        https://pytorch.org/docs/master/optim.html#optimizer-step-closure

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the train_split call.

        """
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            self.optimizer_.zero_grad()
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        if not self.amp_enabled:
            self.optimizer_.step(step_fn)
        else:
            step_fn()  # closure not (yet) supported with AMP
            self.grad_scaler_.step(self.optimizer_)
            self.grad_scaler_.update()

        return step_accumulator.get_step()

    def evaluation_step(self, Xi, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        """
        self.check_is_fitted()
        with torch.set_grad_enabled(training):
            self.module_.train(training)
            return self.infer(Xi)

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        """The proper fit loop.

        Contains the logic of what actually happens during the fit
        loop.

        Parameters
        ----------
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
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        epochs : int or None (default=None)
          If int, train for this number of epochs; if None, use
          ``self.max_epochs``.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(
            X, y, **fit_params)
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(dataset_train, training=True, prefix="train",
                                  step_fn=self.train_step, **fit_params)

            if dataset_valid is not None:
                self.run_single_epoch(dataset_valid, training=False, prefix="valid",
                                      step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        dataset : torch Dataset
            The initialized dataset to loop over.

        training : bool
            Whether to set the module to train mode or not.

        prefix : str
            Prefix to use when saving to the history.

        step_fn : callable
            Function to call for each batch.

        **fit_params : dict
            Additional parameters passed to the ``step_fn``.
        """
        is_placeholder_y = uses_placeholder_y(dataset)

        batch_count = 0
        for data in self.get_iterator(dataset, training=training):
            Xi, yi = unpack_data(data)
            yi_res = yi if not is_placeholder_y else None
            self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            self.history.record_batch(prefix + "_batch_size", get_len(Xi))
            self.notify("on_batch_end", X=Xi, y=yi_res, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    # pylint: disable=unused-argument
    def partial_fit(self, X, y=None, classes=None, **fit_params):
        """Fit the module.

        If the module is initialized, it is not re-initialized, which
        means that this method should be used if you want to continue
        training a model (warm start).

        Parameters
        ----------
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
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        classes : array, sahpe (n_classes,)
          Solely for sklearn compatibility, currently unused.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self

    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Parameters
        ----------
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
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(X, y, **fit_params)
        return self

    def check_is_fitted(self, attributes=None, *args, **kwargs):
        """Checks whether the net is initialized

        Parameters
        ----------
        attributes : iterable of str or None (default=None)
          All the attributes that are strictly required of a fitted
          net. By default, this is the `module_` attribute.

        Other arguments as in
        ``sklearn.utils.validation.check_is_fitted``.

        Raises
        ------
        skorch.exceptions.NotInitializedError
          When the given attributes are not present.

        """
        attributes = attributes or ['module_']
        check_is_fitted(self, attributes, *args, **kwargs)

    def forward_iter(self, X, training=False, device='cpu'):
        """Yield outputs of module forward calls on each batch of data.
        The storage device of the yielded tensors is determined
        by the ``device`` parameter.

        Parameters
        ----------
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

        training : bool (default=False)
          Whether to set the module to train mode or not.

        device : string (default='cpu')
          The device to store each inference result on.
          This defaults to CPU memory since there is genereally
          more memory available there. For performance reasons
          this might be changed to a specific CUDA device,
          e.g. 'cuda:0'.

        Yields
        ------
        yp : torch tensor
          Result from a forward call on an individual batch.

        """
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for data in iterator:
            Xi = unpack_data(data)[0]
            yp = self.evaluation_step(Xi, training=training)
            yield to_device(yp, device=device)

    def forward(self, X, training=False, device='cpu'):
        """Gather and concatenate the output from forward call with
        input data.

        The outputs from ``self.module_.forward`` are gathered on the
        compute device specified by ``device`` and then concatenated
        using PyTorch :func:`~torch.cat`. If multiple outputs are
        returned by ``self.module_.forward``, each one of them must be
        able to be concatenated this way.

        Parameters
        ----------
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

        training : bool (default=False)
          Whether to set the module to train mode or not.

        device : string (default='cpu')
          The device to store each inference result on.
          This defaults to CPU memory since there is genereally
          more memory available there. For performance reasons
          this might be changed to a specific CUDA device,
          e.g. 'cuda:0'.

        Returns
        -------
        y_infer : torch tensor
          The result from the forward step.

        """
        y_infer = list(self.forward_iter(X, training=training, device=device))

        is_multioutput = len(y_infer) > 0 and isinstance(y_infer[0], tuple)
        if is_multioutput:
            return tuple(map(torch.cat, zip(*y_infer)))
        return torch.cat(y_infer)

    @property
    def autocast(self):
        """TODO"""
        return torch.cuda.amp.autocast if self.amp_enabled else no_context

    def _merge_x_and_fit_params(self, x, fit_params):
        duplicates = duplicate_items(x, fit_params)
        if duplicates:
            msg = "X and fit_params contain duplicate keys: "
            msg += ', '.join(duplicates)
            raise ValueError(msg)

        x_dict = dict(x)  # shallow copy
        x_dict.update(fit_params)
        return x_dict

    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        Parameters
        ----------
        x : input data
          A batch of the input data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        x = to_tensor(x, device=self.device)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        with self.autocast():
            return self.module_(x, **fit_params)

    def _get_predict_nonlinearity(self):
        """Return the nonlinearity to be applied to the prediction

        This can be useful, e.g., when
        :func:`~skorch.classifier.NeuralNetClassifier.predict_proba`
        should return probabilities but a criterion is used that does
        not expect probabilities. In that case, the module can return
        whatever is required by the criterion and the
        ``predict_nonlinearity`` transforms this output into
        probabilities.

        The nonlinearity is applied only when calling
        :func:`~skorch.classifier.NeuralNetClassifier.predict` or
        :func:`~skorch.classifier.NeuralNetClassifier.predict_proba`
        but not anywhere else -- notably, the loss is unaffected by
        this nonlinearity.

        Raises
        ------
        TypeError
          Raise a TypeError if the return value is not callable.

        Returns
        -------
        nonlin : callable
          A callable that takes a single argument, which is a PyTorch
          tensor, and returns a PyTorch tensor.

        """
        self.check_is_fitted()
        nonlin = self.predict_nonlinearity
        if nonlin is None:
            nonlin = _identity
        elif nonlin == 'auto':
            nonlin = _infer_predict_nonlinearity(self)
        if not callable(nonlin):
            raise TypeError("predict_nonlinearity has to be a callable, 'auto' or None")
        return nonlin

    def predict_proba(self, X):
        """Return the output of the module's forward method as a numpy
        array.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
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

        Returns
        -------
        y_proba : numpy ndarray

        """
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict(self, X):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
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

        Returns
        -------
        y_pred : numpy ndarray

        """
        return self.predict_proba(X)

    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values

        y_true : torch tensor
          True target values.

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

        training : bool (default=False)
          Whether train mode should be used or not.

        """
        y_true = to_tensor(y_true, device=self.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        with self.autocast():
            loss = self.criterion_(y_pred, y_true)
        return self.grad_scaler_.scale(loss) if self.amp_enabled else loss

    def get_dataset(self, X, y=None):
        """Get a dataset that contains the input data and is passed to
        the iterator.

        Override this if you want to initialize your dataset
        differently.

        Parameters
        ----------
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
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        Returns
        -------
        dataset
          The initialized dataset.

        """
        if is_dataset(X):
            return X

        dataset = self.dataset
        is_initialized = not callable(dataset)

        kwargs = self.get_params_for('dataset')
        if kwargs and is_initialized:
            raise TypeError("Trying to pass an initialized Dataset while "
                            "passing Dataset arguments ({}) is not "
                            "allowed.".format(kwargs))

        if is_initialized:
            return dataset

        return dataset(X, y, **kwargs)

    def get_split_datasets(self, X, y=None, **fit_params):
        """Get internal train and validation datasets.

        The validation dataset can be None if ``self.train_split`` is
        set to None; then internal validation will be skipped.

        Override this if you want to change how the net splits
        incoming data into train and validation part.

        Parameters
        ----------
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
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          Additional parameters passed to the ``self.train_split``
          call.

        Returns
        -------
        dataset_train
          The initialized training dataset.

        dataset_valid
          The initialized validation dataset or None

        """
        dataset = self.get_dataset(X, y)
        if not self.train_split:
            return dataset, None

        # After a change in (#646),
        # `y` is no longer passed to `self.train_split` if it is `None`.
        # To revert to the previous behavior, remove the following two lines:
        if y is None:
            return self.train_split(dataset, **fit_params)

        return self.train_split(dataset, y, **fit_params)

    def get_iterator(self, dataset, training=False):
        """Get an iterator that allows to loop over the batches of the
        given data.

        If ``self.iterator_train__batch_size`` and/or
        ``self.iterator_test__batch_size`` are not set, use
        ``self.batch_size`` instead.

        Parameters
        ----------
        dataset : torch Dataset (default=skorch.dataset.Dataset)
          Usually, ``self.dataset``, initialized with the corresponding
          data, is passed to ``get_iterator``.

        training : bool (default=False)
          Whether to use ``iterator_train`` or ``iterator_test``.

        Returns
        -------
        iterator
          An instantiated iterator that allows to loop over the
          mini-batches.

        """
        if training:
            kwargs = self.get_params_for('iterator_train')
            iterator = self.iterator_train
        else:
            kwargs = self.get_params_for('iterator_valid')
            iterator = self.iterator_valid

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        if kwargs['batch_size'] == -1:
            kwargs['batch_size'] = len(dataset)

        return iterator(dataset, **kwargs)

    def _get_params_for(self, prefix):
        return params_for(prefix, self.__dict__)

    def get_params_for(self, prefix):
        """Collect and return init parameters for an attribute.

        Attributes could be, for instance, pytorch modules, criteria,
        or data loaders (for optimizers, use
        :meth:`.get_params_for_optimizer` instead). Use the returned
        arguments to initialize the given attribute like this:

        .. code:: python

            # inside initialize_module method
            kwargs = self.get_params_for('module')
            self.module_ = self.module(**kwargs)

        Proceed analogously for the criterion etc.

        The reason to use this method is so that it's possible to
        change the init parameters with :meth:`.set_params`, which
        in turn makes grid search and other similar things work.

        Note that in general, as a user, you never have to deal with
        this method because :meth:`.initialize_module` etc. are
        already taking care of this. You only need to deal with this
        if you override :meth:`.initialize_module` (or similar
        methods) because you have some custom code that requires it.

        Parameters
        ----------
        prefix : str
          The name of the attribute whose arguments should be
          returned. E.g. for the module, it should be ``'module'``.

        Returns
        -------
        kwargs : dict
          Keyword arguments to be used as init parameters.

        """
        return self._get_params_for(prefix)

    def _get_params_for_optimizer(self, prefix, named_parameters):
        kwargs = self.get_params_for(prefix)
        params = list(named_parameters)
        pgroups = []

        for pattern, group in kwargs.pop('param_groups', []):
            matches = [i for i, (name, _) in enumerate(params) if
                       fnmatch.fnmatch(name, pattern)]
            if matches:
                p = [params.pop(i)[1] for i in reversed(matches)]
                pgroups.append({'params': p, **group})

        if params:
            pgroups.append({'params': [p for _, p in params]})

        args = (pgroups,)
        return args, kwargs

    def get_params_for_optimizer(self, prefix, named_parameters):
        """Collect and return init parameters for an optimizer.

        Parse kwargs configuration for the optimizer identified by
        the given prefix. Supports param group assignment using wildcards:

        .. code:: python

            optimizer__lr=0.05,
            optimizer__param_groups=[
                ('rnn*.period', {'lr': 0.3, 'momentum': 0}),
                ('rnn0', {'lr': 0.1}),
            ]

        Generally, use this method like this:

        .. code:: python

            # inside initialize_optimizer method
            named_params = self.module_.named_parameters()
            pgroups, kwargs = self.get_params_for_optimizer('optimizer', named_params)
            if 'lr' not in kwargs:
                kwargs['lr'] = self.lr
            self.optimizer_ = self.optimizer(*pgroups, **kwargs)

        The reason to use this method is so that it's possible to
        change the init parameters with :meth:`.set_params`, which
        in turn makes grid search and other similar things work.

        Note that in general, as a user, you never have to deal with
        this method because :meth:`.initialize_optimizer` is already
        taking care of this. You only need to deal with this if you
        override :meth:`.initialize_optimizer` because you have some
        custom code that requires it.

        Parameters
        ----------
        prefix : str
          The name of the optimizer whose arguments should be
          returned. Typically, this should just be
          ``'optimizer'``. There can be exceptions, however, e.g. if
          you want to use more than one optimizer.

        named_parameters : iterator
          Iterator over the parameters of the module that is intended
          to be optimized. It's the return value of
          ``my_module.named_parameters()``.

        Returns
        -------
        args : tuple
          All positional arguments for this optimizer (right now only
          one, the parameter groups).

        kwargs : dict
          All other parameters for this optimizer, e.g. the learning
          rate.

        """
        args, kwargs = self._get_params_for_optimizer(prefix, named_parameters)
        return args, kwargs

    def _get_param_names(self):
        return (k for k in self.__dict__ if not k.endswith('_'))

    def _get_params_callbacks(self, deep=True):
        """sklearn's .get_params checks for `hasattr(value,
        'get_params')`. This returns False for a list. But our
        callbacks reside within a list. Hence their parameters have to
        be retrieved separately.
        """
        params = {}
        if not deep:
            return params

        callbacks_ = getattr(self, 'callbacks_', [])
        for key, val in chain(callbacks_, self._default_callbacks):
            name = 'callbacks__' + key
            params[name] = val
            if val is None:  # callback deactivated
                continue
            for subkey, subval in val.get_params().items():
                subname = name + '__' + subkey
                params[subname] = subval
        return params

    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        # Callback parameters are not returned by .get_params, needs
        # special treatment.
        params_cb = self._get_params_callbacks(deep=deep)
        params.update(params_cb)
        return params

    def _check_kwargs(self, kwargs):
        """Check argument names passed at initialization.

        Raises
        ------
        TypeError
          Raises a TypeError if one or more arguments don't seem to
          match or are malformed.

        Returns
        -------
        kwargs: dict
          Return the passed keyword arguments.

        Example
        -------
        >>> net = NeuralNetClassifier(MyModule, iterator_train_shuffle=True)
        TypeError: Got an unexpected argument iterator_train_shuffle,
        did you mean iterator_train__shuffle?

        """
        # warn about usage of iterator_valid__shuffle=True, since this
        # is almost certainly not what the user wants
        if kwargs.get('iterator_valid__shuffle'):
            warnings.warn(
                "You set iterator_valid__shuffle=True; this is most likely not "
                "what you want because the values returned by predict and "
                "predict_proba will be shuffled.",
                UserWarning)

        # check for wrong arguments
        unexpected_kwargs = []
        missing_dunder_kwargs = []
        for key in kwargs:
            if key.endswith('_'):
                continue

            # see https://github.com/skorch-dev/skorch/pull/590 for
            # why this must be sorted
            for prefix in sorted(self.prefixes_, key=lambda s: (-len(s), s)):
                if key.startswith(prefix):
                    if not key.startswith(prefix + '__'):
                        missing_dunder_kwargs.append((prefix, key))
                    break
            else:  # no break means key didn't match a prefix
                unexpected_kwargs.append(key)

        msgs = []
        if unexpected_kwargs:
            tmpl = ("__init__() got unexpected argument(s) {}. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
            msg = tmpl.format(', '.join(sorted(unexpected_kwargs)))
            msgs.append(msg)

        for prefix, key in sorted(missing_dunder_kwargs, key=lambda tup: tup[1]):
            tmpl = "Got an unexpected argument {}, did you mean {}?"
            suffix = key[len(prefix):].lstrip('_')
            suggestion = prefix + '__' + suffix
            msgs.append(tmpl.format(key, suggestion))

        if msgs:
            full_msg = '\n'.join(msgs)
            raise TypeError(full_msg)

        return kwargs

    def _check_deprecated_params(self, **kwargs):
        pass

    def set_params(self, **kwargs):
        """Set the parameters of this class.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self

        """
        self._check_deprecated_params(**kwargs)
        normal_params, cb_params, special_params = {}, {}, {}
        virtual_params = {}

        for key, val in kwargs.items():
            if self._is_virtual_param(key):
                virtual_params[key] = val
            elif key.startswith('callbacks'):
                cb_params[key] = val
            elif any(key.startswith(prefix) for prefix in self.prefixes_):
                special_params[key] = val
            else:
                normal_params[key] = val

        self._apply_virtual_params(virtual_params)
        BaseEstimator.set_params(self, **normal_params)

        for key, val in special_params.items():
            if key.endswith('_'):
                raise ValueError(
                    "Something went wrong here. Please open an issue on "
                    "https://github.com/skorch-dev/skorch/issues detailing what "
                    "caused this error.")
            setattr(self, key, val)

        # Below: Re-initialize parts of the net if necessary.

        if cb_params:
            # callbacks need special treatmeant since they are list of tuples
            self.initialize_callbacks()
            self._set_params_callback(**cb_params)

        if any('criterion' in key.split('__', 1)[0] for key in special_params):
            self.initialize_criterion()

        module_triggers_optimizer_reinit = False
        if any('module' in key.split('__', 1)[0] for key in special_params):
            self.initialize_module()
            module_triggers_optimizer_reinit = True

        optimizer_changed = (
            any('optimizer' in key.split('__', 1)[0] for key in special_params)
            or 'lr' in normal_params
        )
        if module_triggers_optimizer_reinit or optimizer_changed:
            # Model selectors such as GridSearchCV will set the
            # parameters before .initialize() is called, therefore we
            # need to make sure that we have an initialized model here
            # as the optimizer depends on it.
            if not hasattr(self, 'module_'):
                self.initialize_module()

            # If we reached this point but the optimizer was not
            # changed, it means that optimizer initialization was
            # triggered indirectly.
            self.initialize_optimizer(triggered_directly=optimizer_changed)

        vars(self).update(kwargs)

        return self

    def _set_params_callback(self, **params):
        """Special handling for setting params on callbacks."""
        # model after sklearn.utils._BaseCompostion._set_params
        # 1. All steps
        if 'callbacks' in params:
            setattr(self, 'callbacks', params.pop('callbacks'))

        # 2. Step replacement
        names, _ = zip(*getattr(self, 'callbacks_'))
        for key in params.copy():
            name = key[11:]  # drop 'callbacks__'
            if '__' not in name and name in names:
                self._replace_callback(name, params.pop(key))

        # 3. Step parameters and other initilisation arguments
        for key in params.copy():
            name = key[11:]
            part0, part1 = name.split('__')
            kwarg = {part1: params.pop(key)}
            callback = dict(self.callbacks_).get(part0)
            if callback is not None:
                callback.set_params(**kwarg)
            else:
                raise ValueError(
                    "Trying to set a parameter for callback {} "
                    "which does not exist.".format(part0))

        return self

    def _replace_callback(self, name, new_val):
        # assumes `name` is a valid callback name
        callbacks_new = self.callbacks_[:]
        for i, (cb_name, _) in enumerate(callbacks_new):
            if cb_name == name:
                callbacks_new[i] = (name, new_val)
                break
        setattr(self, 'callbacks_', callbacks_new)

    def __getstate__(self):
        state = self.__dict__.copy()
        cuda_attrs = {}
        for prefix in self.cuda_dependent_attributes_:
            for key, val in state.items():
                key_matches = isinstance(key, str) and key.startswith(prefix)
                if key_matches and (val is not None):
                    cuda_attrs[key] = state[key]

        for k in cuda_attrs:
            state.pop(k)

        with tempfile.SpooledTemporaryFile() as f:
            torch.save(cuda_attrs, f)
            f.seek(0)
            state['__cuda_dependent_attributes__'] = f.read()

        return state

    def __setstate__(self, state):
        # get_map_location will automatically choose the
        # right device in cases where CUDA is not available.
        map_location = get_map_location(state['device'])
        load_kwargs = {'map_location': map_location}
        state['device'] = self._check_device(state['device'], map_location)

        with tempfile.SpooledTemporaryFile() as f:
            f.write(state['__cuda_dependent_attributes__'])
            f.seek(0)
            cuda_attrs = torch.load(f, **load_kwargs)

        state.update(cuda_attrs)
        state.pop('__cuda_dependent_attributes__')

        self.__dict__.update(state)

    def _register_attribute(
            self,
            name,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """Add attribute name to prefixes_ and
        cuda_dependent_attributes_.

        The first is to take care that the attribute works correctly
        with set_params, e.g. when it comes to re-initialization.

        The second is to make sure that nets trained with CUDA can be
        loaded without CUDA.

        This method takes care of not mutating the lists.

        Parameters
        ----------
        prefixes : bool (default=True)
          Whether to add to prefixes_.

        cuda_dependent_attributes : bool (default=True)
          Whether to add to cuda_dependent_attributes_.

        """
        # copy the lists to avoid mutation
        if prefixes:
            self.prefixes_ = self.prefixes_[:] + [name]

        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = (
                self.cuda_dependent_attributes_[:] + [name + '_'])

    def _unregister_attribute(
            self,
            name,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """Remove attribute name from prefixes_ and
        cuda_dependent_attributes_.

        Use this to remove PyTorch components that are not needed
        anymore. This is mostly a clean up job, so as to not leave
        unnecessary prefixes or cuda-dependent attributes.

        This method takes care of not mutating the lists.

        Parameters
        ----------
        prefixes : bool (default=True)
          Whether to remove from prefixes_.

        cuda_dependent_attributes : bool (default=True)
          Whether to remove from cuda_dependent_attributes_.

        """
        # copy the lists to avoid mutation
        if prefixes:
            self.prefixes_ = [p for p in self.prefixes_ if p != name]

        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = [
                a for a in self.cuda_dependent_attributes_ if a != name + '_']

    def __setattr__(self, name, attr):
        """Set an attribute on the net

        When a custom net with additional torch modules or optimizers
        is created, those attributes are added to ``prefixes_`` and
        ``cuda_dependent_attributes_`` automatically.

        """
        # If it's a
        # 1. known attribute or
        # 2. special param like module__num_units or
        # 3. not a torch module/optimizer instance or class
        # just setattr as usual.
        # For a discussion why we chose this implementation, see here:
        # https://github.com/skorch-dev/skorch/pull/597
        is_known = name.endswith('_') or (name in self.prefixes_)
        is_special_param = '__' in name
        is_torch_component = any(c in name for c in _PYTORCH_COMPONENTS)

        if not (is_known or is_special_param) and is_torch_component:
            self._register_attribute(name)
        super().__setattr__(name, attr)

    def __delattr__(self, name):
        # take extra precautions to undo the changes made in __setattr__
        self._unregister_attribute(name)
        super().__delattr__(name)

    def _get_module(self, name, msg):
        """Return the PyTorch module with the given name

        If a module with such a name doesn't exist, also check the
        name without trailing underscore.

        Raises
        ------
        AttributeError
          If no module with the given name could be found, with a
          message formatted according to the passed ``msg`` argument.

        """
        try:
            module = getattr(self, name)
            if not hasattr(module, 'state_dict'):
                raise AttributeError
            return module
        except AttributeError:
            if name.endswith('_'):
                name = name[:-1]
            raise AttributeError(msg.format(name=name))

    def save_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            f_grad_scaler=None,
            **kwargs):
        """Saves the module's parameters, history, and optimizer,
        not the whole object.

        To save the whole object, use pickle. This is necessary when
        you need additional learned attributes on the net, e.g. the
        ``classes_`` attribute on
        :class:`skorch.classifier.NeuralNetClassifier`.

        ``f_params``, ``f_optimizer``, etc. use PyTorch's
        :func:`~torch.save`.

        If you've created a custom module, e.g. ``net.mymodule_``, you
        can save that as well by passing ``f_mymodule``.

        Parameters
        ----------
        f_params : file-like object, str, None (default=None)
          Path of module parameters. Pass ``None`` to not save

        f_optimizer : file-like object, str, None (default=None)
          Path of optimizer. Pass ``None`` to not save

        f_criterion : file-like object, str, None (default=None)
          Path of criterion. Pass ``None`` to not save

        f_history : file-like object, str, None (default=None)
          Path to history. Pass ``None`` to not save

        f_grad_scaler TODO

        Examples
        --------
        >>> before = NeuralNetClassifier(mymodule)
        >>> before.save_params(f_params='model.pkl',
        ...                    f_optimizer='optimizer.pkl',
        ...                    f_history='history.json')
        >>> after = NeuralNetClassifier(mymodule).initialize()
        >>> after.load_params(f_params='model.pkl',
        ...                   f_optimizer='optimizer.pkl',
        ...                   f_history='history.json')

        """
        kwargs_module, kwargs_other = _check_f_arguments(
            'save_params',
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_criterion=f_criterion,
            f_history=f_history,
            f_grad_scaler=f_grad_scaler,
            **kwargs)

        if not kwargs_module and not kwargs_other:
            print("Nothing to save")
            return

        msg_init = (
            "Cannot save state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...).")
        msg_module = (
            "You are trying to save 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module; make sure that it exists and check for typos.")

        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr[:-1] in self.prefixes_:
                self.check_is_fitted([attr], msg=msg_init)
            module = self._get_module(attr, msg=msg_module)
            torch.save(module.state_dict(), f_name)

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history.to_file(f_history)

    def _check_device(self, requested_device, map_device):
        """Compare the requested device with the map device and
        return the map device if it differs from the requested device
        along with a warning.
        """
        type_1 = torch.device(requested_device)
        type_2 = torch.device(map_device)
        if type_1 != type_2:
            warnings.warn(
                'Setting self.device = {} since the requested device ({}) '
                'is not available.'.format(map_device, requested_device),
                DeviceWarning)
            return map_device
        # return requested_device instead of map_device even though we
        # checked for *type* equality as we might have 'cuda:0' vs. 'cuda:1'.
        return requested_device

    def load_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            f_grad_scaler=None,
            checkpoint=None,
            **kwargs):
        """Loads the the module's parameters, history, and optimizer,
        not the whole object.

        To save and load the whole object, use pickle.

        ``f_params``, ``f_optimizer``, etc. uses PyTorch's
        :func:`~torch.load`.

        If you've created a custom module, e.g. ``net.mymodule_``, you
        can save that as well by passing ``f_mymodule``.

        Parameters
        ----------
        f_params : file-like object, str, None (default=None)
          Path of module parameters. Pass ``None`` to not load.

        f_optimizer : file-like object, str, None (default=None)
          Path of optimizer. Pass ``None`` to not load.

        f_criterion : file-like object, str, None (default=None)
          Path of criterion. Pass ``None`` to not save

        f_history : file-like object, str, None (default=None)
          Path to history. Pass ``None`` to not load.

        checkpoint : :class:`.Checkpoint`, None (default=None)
          Checkpoint to load params from. If a checkpoint and a ``f_*``
          path is passed in, the ``f_*`` will be loaded. Pass
          ``None`` to not load.

        Examples
        --------
        >>> before = NeuralNetClassifier(mymodule)
        >>> before.save_params(f_params='model.pkl',
        >>>                    f_optimizer='optimizer.pkl',
        >>>                    f_history='history.json')
        >>> after = NeuralNetClassifier(mymodule).initialize()
        >>> after.load_params(f_params='model.pkl',
        >>>                   f_optimizer='optimizer.pkl',
        >>>                   f_history='history.json')

        """
        def _get_state_dict(f):
            map_location = get_map_location(self.device)
            self.device = self._check_device(self.device, map_location)
            return torch.load(f, map_location=map_location)

        kwargs_full = {}
        if checkpoint is not None:
            if not self.initialized_:
                self.initialize()
            if f_history is None and checkpoint.f_history is not None:
                self.history = History.from_file(checkpoint.f_history_)
            kwargs_full.update(**checkpoint.get_formatted_files(self))

        # explicit arguments may override checkpoint arguments
        kwargs_full.update(**kwargs)
        for key, val in [('f_params', f_params), ('f_optimizer', f_optimizer),
                         ('f_criterion', f_criterion), ('f_history', f_history)]:
            if val:
                kwargs_full[key] = val

        kwargs_module, kwargs_other = _check_f_arguments('load_params', **kwargs_full)

        if not kwargs_module and not kwargs_other:
            print("Nothing to load")
            return

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history = History.from_file(f_history)

        msg_init = (
            "Cannot load state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...).")
        msg_module = (
            "You are trying to load 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module; make sure that it exists and check for typos.")

        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr[:-1] in self.prefixes_:
                self.check_is_fitted([attr], msg=msg_init)
            module = self._get_module(attr, msg=msg_module)
            state_dict = _get_state_dict(f_name)
            module.load_state_dict(state_dict)

    def __repr__(self):
        to_include = ['module']
        to_exclude = []
        parts = [str(self.__class__) + '[uninitialized](']
        if self.initialized_:
            parts = [str(self.__class__) + '[initialized](']
            to_include = ['module_']
            to_exclude = ['module__']

        for key, val in sorted(self.__dict__.items()):
            if not any(key.startswith(prefix) for prefix in to_include):
                continue
            if any(key.startswith(prefix) for prefix in to_exclude):
                continue

            val = str(val)
            if '\n' in val:
                val = '\n  '.join(val.split('\n'))
            parts.append('  {}={},'.format(key, val))

        parts.append(')')
        return '\n'.join(parts)
