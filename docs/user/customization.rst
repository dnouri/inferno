=============
Customization
=============

Customizing NeuralNet
---------------------

Apart from the :class:`.NeuralNet` base class, we provide
:class:`.NeuralNetClassifier`, :class:`.NeuralNetBinaryClassifier`,
and :class:`.NeuralNetRegressor` for typical classification, binary
classification, and regressions tasks. They should work as drop-in
replacements for sklearn classifiers and regressors.

The :class:`.NeuralNet` class is a little less opinionated about the
incoming data, e.g. it does not determine a loss function by default.
Therefore, if you want to write your own subclass for a special use
case, you would typically subclass from :class:`.NeuralNet`. The
:func:`~skorch.net.NeuralNet.predict` method returns the same output
as :func:`~skorch.net.NeuralNet.predict_proba` by default, which is
the module output (or the first module output, in case it returns
multiple values).

:class:`.NeuralNet` and its subclasses are already very flexible as they are and
should cover many use cases by adjusting the provided parameters or by using
callbacks. However, this may not always be sufficient for your use cases. If you
thus find yourself wanting to customize :class:`.NeuralNet`, please follow the
guidelines in this section.

Methods starting with get_*
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The net provides a few ``get_*`` methods, most notably
:func:`~skorch.net.NeuralNet.get_loss`,
:func:`~skorch.net.NeuralNet.get_dataset`, and
:func:`~skorch.net.NeuralNet.get_iterator`. The intent of these
methods should be pretty self-explanatory, and if you are still not
quite sure, consult their documentations. In general, these methods
are fairly safe to override as long as you make sure to conform to the
same signature as the original.

A short example should serve to illustrate this.
:func:`~skorch.net.NeuralNet.get_loss` is called when the loss is determined.
Below we show an example of overriding :func:`~skorch.net.NeuralNet.get_loss` to
add L1 regularization to our total loss:

.. code:: python

    class RegularizedNet(NeuralNet):
        def __init__(self, *args, lambda1=0.01, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda1 = lambda1

        def get_loss(self, y_pred, y_true, X=None, training=False):
            loss = super().get_loss(y_pred, y_true, X=X, training=training)
            loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
            return loss

.. note:: This example also regularizes the biases, which you typically
    don't need to do.

It is often a good idea to call ``super`` of the method you override, to make
sure that everything that needs to happen inside that method does happen. If you
don't, you should make sure to take care of everything that needs to happen by
following the original implementation.

Training and validation
^^^^^^^^^^^^^^^^^^^^^^^

If you would like to customize training and validation, there are
several possibilities. Below are the methods that you most likely want
to customize:

The method :func:`~skorch.net.NeuralNet.train_step_single` performs a
single training step. It accepts the current batch of data as input
(as well as the ``fit_params``) and should return a dictionary
containing the ``loss`` and the prediction ``y_pred``. E.g. you should
override this if your dataset returns some non-standard data that
needs custom handling, and/or if your module has to be called in a
very specific way. If you want to, you can still make use of
:func:`~skorch.net.NeuralNet.infer` and
:func:`~skorch.net.NeuralNet.get_loss` but it's not strictly
necessary. Don't call the optimizer in this method, this is handled by
the next method.

The method :func:`~skorch.net.NeuralNet.train_step` defines the
complete training procedure performed for each batch. It accepts the
same arguments as :func:`~skorch.net.NeuralNet.train_step_single` but
it differs in that it defines the training closure passed to the
optimizer, which for instance could be called more than once (e.g. in
case of L-BFGS). You might override this if you deal with non-standard
training procedures, as e.g. gradient accumulation.

The method :func:`~skorch.net.NeuralNet.validation_step` is
responsible for calculating the prediction and loss on the validation
data (remember that skorch uses an internal validation set for
reporting, early stopping, etc.). Similar to
:func:`~skorch.net.NeuralNet.train_step_single`, it receives the batch
and ``fit_params`` as input and should return a dictionary containing
``loss`` and ``y_pred``. Most likely, when you need to customize
:func:`~skorch.net.NeuralNet.train_step_single`, you'll need to
customize :func:`~skorch.net.NeuralNet.validation_step` accordingly.

Finally, the method :func:`~skorch.net.NeuralNet.evaluation_step` is
called to you use the net for inference, e.g. when calling
:func:`~skorch.net.NeuralNet.forward` or
:func:`~skorch.net.NeuralNet.predict`. You may want to modify this if,
e.g., you want your model to behave differently during training and
during prediction.

You should also be aware that some methods are better left
untouched. E.g., in most cases, the following methods should *not* be
overridden:

* :func:`~skorch.net.NeuralNet.fit`
* :func:`~skorch.net.NeuralNet.partial_fit`
* :func:`~skorch.net.NeuralNet.fit_loop`
* :func:`~skorch.net.NeuralNet.run_single_epoch`

The reason why these methods should stay untouched is because they
perform some book keeping, like making sure that callbacks are handled
or writing logs to the ``history``. If you do need to override these,
make sure that you perform the same book keeping as the original
methods.

Initialization and custom modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method :func:`~skorch.net.NeuralNet.initialize` is responsible for
initializing all the components needed by the net, e.g. the module and
the optimizer. For this, it calls specific initialization methods,
such as :func:`~skorch.net.NeuralNet.initialize_module` and
:func:`~skorch.net.NeuralNet.initialize_optimizer`. If you'd like to
customize the initialization behavior, you should override the
corresponding methods. Following sklearn conventions, the created
components should be set as an attribute with a trailing underscore as
the name, e.g. ``module_`` for the initialized module.

A possible modification you may want to make is to add more modules, criteria,
and optimizers to your net. This is possible in skorch by following the
guidelines below. If you do this, your custom modules and optimizers will be
treated as "first class citizens" in skorch land. This means:

1. The parameters of your custom modules are automatically passed to the
   optimizer (but you can modify this behavior).
2. skorch takes care of moving your modules to the correct device.
3. skorch takes care of setting the training/eval mode correctly.
4. When a module needs to be re-initialized because ``set_params`` was called,
   all modules and optimizers that may depend on it are also re-initialized.
   This is for instance important for the optimizer, which must know about the
   parameters of the newly initialized module.
5. You can pass arguments to the custom modules and optimizers using the now
   familiar double-underscore notation. E.g., you can initialize your net like
   this:

.. code:: python

    net = MyNet(
        module=MyModule,
        module__num_units=100,

        othermodule=MyOtherModule,
        othermodule__num_units=200,
    )
    net.fit(X, y)

A word about the distinction between modules and criteria made by skorch:
Typically, criteria are also just subclasses of PyTorch
:class:`~torch.nn.Module`. As such, skorch moves them to CUDA if that is the
indicated device and will even pass parameters of criteria to the optimizers, if
there are any. This can be useful when e.g. training GANs, where you might
implement the discriminator as the criterion (and the generator as the module).

A difference between module and criterion is that the output of modules are used
for generating the predictions and are thus returned by
:func:`~skorch.net.NeuralNet.predict` etc. In contrast, the output of the
criterion is used for calculating the loss and should therefore be a scalar.

skorch assumes that criteria may depend on the modules. Therefore, if a module
is re-initialized, all criteria are also re-initialized, but not vice-versa. On
top of that, the optimizer is re-initialized when either modules or criteria
are changed.

So after all this talk, what are the aforementioned guidelines to add your own
modules, criteria, and optimizers? You have to follow these rules:

1. Initialize them during their respective ``initialize_`` methods, e.g. modules
   should be set inside :func:`~skorch.net.NeuralNet.initialize_module`.
2. If they have learnable parameters, they should be instances of
   :class:`~torch.nn.Module`. Optimizers should be instances of
   :class:`~torch.optim.Optimizer`.
3. Their names should end on an underscore. This is true for all attributes that
   are created during ``initialize`` and distinguishes them from arguments
   passed to ``__init__``. So a name for a custom module could be ``mymodule_``.
4. Inside the initialization method, use :meth:`.get_params_for` (or,
   if dealing with an optimizer, :meth:`.get_params_for_optimizer`) to
   retrieve the arguments for the constructor of the instance.

Here is an example of how this could look like in practice:

.. code:: python

    class MyNet(NeuralNet):
        def initialize_module(self, *args, **kwargs):
            super().initialize_module(*args, **kwargs)

            # add an additional module called 'module2_'
            params = self.get_params_for('module2')
            self.module2_ = Module2(**params)
            return self

        def initialize_criterion(self, *args, **kwargs):
            super().initialize_criterion(*args, **kwargs)

            # add an additional criterion called 'other_criterion_'
            params = self.get_params_for('other_criterion')
            self.other_criterion_ = nn.BCELoss(**params)
            return self

        def initialize_optimizer(self, *args, **kwargs):
            # first initialize the normal optimizer
            named_params = self.module_.named_parameters()
            args, kwargs = self.get_params_for_optimizer('optimizer', named_params)
            self.optimizer_ = self.optimizer(*args, **kwargs)

            # next add an another optimizer called 'optimizer2_' that is
            # only responsible for training 'module2_'
            named_params = self.module2_.named_parameters()
            args, kwargs = self.get_params_for_optimizer('optimizer2', named_params)
            self.optimizer2_ = torch.optim.SGD(*args, **kwargs)
            return self

        ...  # additional changes


    net = MyNet(
        ...,
        module2__num_units=123,
        other_criterion__reduction='sum',
        optimizer2__lr=0.1,
    )
    net.fit(X, y)

    # set_params works
    net.set_params(optimizer2__lr=0.05)
    net.partial_fit(X, y)

    # grid search et al. works
    search = GridSearchCV(net, {'module2__num_units': [10, 50, 100]}, ...)
    search.fit(X, y)

In this example, a new criterion, a new module, and a new optimizer
were added. Of course, additional changes should be made to the net so
that those new components are actually being used for something, but
this example should illustrate how to start. Since the rules outlined
above are being followed, we can use grid search on our customly
defined components.

.. note:: In the example above, the parameters of ``module_`` are trained by
          ``optimzer_`` and the parameters of ``module2_`` are trained by
          ``optimizer2_``. To conveniently obtain the parameters of all modules,
          call the method :func:`~skorch.net.NeuralNet.get_all_learnable_params`.
