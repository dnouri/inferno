===
FAQ
===

How do I apply L2 regularization?
---------------------------------

To apply L2 regularization (aka weight decay), PyTorch supplies
the ``weight_decay`` parameter, which must be supplied to the
optimizer. To pass this variable in skorch, use the
double-underscore notation for the optimizer:

.. code:: python

    net = NeuralNet(
        ...,
        optimizer__weight_decay=0.01,
    )


How can I continue training my model?
-------------------------------------

By default, when you call :func:`~skorch.net.NeuralNet.fit` more than
once, the training starts from zero instead of from where it was left.
This is in line with sklearn\'s behavior but not always desired. If
you would like to continue training, use
:func:`~skorch.net.NeuralNet.partial_fit` instead of
:func:`~skorch.net.NeuralNet.fit`. Alternatively, there is the
``warm_start`` argument, which is ``False`` by default. Set it to
``True`` instead and you should be fine.


How do I shuffle my train batches?
----------------------------------

skorch uses :class:`~torch.utils.data.DataLoader` from PyTorch under
the hood.  This class takes a couple of arguments, for instance
``shuffle``. We therefore need to pass the ``shuffle`` argument to
:class:`~torch.utils.data.DataLoader`, which we achieve by using the
double-underscore notation (as known from sklearn):

.. code:: python

    net = NeuralNet(
        ...,
        iterator_train__shuffle=True,
    )

Note that we have an ``iterator_train`` for the training data and an
``iterator_valid`` for validation and test data. In general, you only
want to shuffle the train data, which is what the code above does.


How do I use sklearn GridSeachCV when my data is in a dictionary?
-----------------------------------------------------------------

skorch supports dicts as input but sklearn doesn't. To get around
that, try to wrap your dictionary into a :class:`.SliceDict`. This is
a data container that partly behaves like a dict, partly like an
ndarray. For more details on how to do this, have a look at the
coresponding `data section
<https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Working-with-sklearn-FunctionTransformer-and-GridSearch>`__
in the notebook.


How do I use sklearn GridSeachCV when my data is in a dataset?
-----------------------------------------------------------------

skorch supports datasets as input but sklearn doesn't. If it's
possible, you should provide your data in a non-dataset format,
e.g. as a numpy array or torch tensor, extracted from your original
dataset.

Sometimes, this is not possible, e.g. when your data doesn't fit into
memory. To get around that, try to wrap your dataset into a
:class:`.SliceDataset`. This is a data container that partly behaves
like a dataset, partly like an ndarray. Further information can be
found here: :ref:`slicedataset`.

I want to use sample_weight, how can I do this?
-----------------------------------------------

Some scikit-learn models support to pass a ``sample_weight`` argument
to ``fit`` calls as part of the ``fit_params``. This allows you to
give different samples different weights in the final loss
calculation.

In general, skorch supports ``fit_params``, but unfortunately just
calling ``net.fit(X, y, sample_weight=sample_weight)`` is not enough,
because the ``fit_params`` are not split into train and valid, and are
not batched, resulting in a mismatch with the training batches.

Fortunately, skorch supports passing dictionaries as arguments, which
are actually split into train and valid and then batched. Therefore,
the best solution is to pass the ``sample_weight`` with ``X`` as a
dictionary. Below, there is example code on how to achieve this:

.. code:: python

    X, y = get_data()
    # put your X into a dict if not already a dict
    X = {'data': X}
    # add sample_weight to the X dict
    X['sample_weight'] = sample_weight

    class MyModule(nn.Module):
        ...  # normal PyTorch module

    class MyNet(NeuralNet):
        def __init__(self, *args, **kwargs):
            # make sure to set reduce=False in your criterion
            # (since we need the loss for each sample so that it can be weighted)
            super().__init__(*args, criterion__reduce=False, **kwargs)

        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # Pass the data to get unnormalized loss
            loss_unreduced = super().get_loss(y_pred, y_true, X["data"], *args, **kwargs)

            # Reweight the loss with sample_weight
            loss_reduced = (X["sample_weight"] * loss_unreduced).mean()
            return loss_reduced

    net = MyNet(MyModule, ...)
    net.fit(X, y)


I already split my data into training and validation sets, how can I use them?
------------------------------------------------------------------------------

If you have predefined training and validation datasets that are
subclasses of PyTorch :class:`~torch.utils.data.Dataset`, you can use
:func:`~skorch.helper.predefined_split` to wrap your validation dataset and
pass it to :class:`~skorch.net.NeuralNet`'s ``train_split`` parameter:

.. code:: python

    from skorch.helper import predefined_split

    net = NeuralNet(
        ...,
        train_split=predefined_split(valid_ds)
    )
    net.fit(train_ds)

If you split your data by using :func:`~sklearn.model_selection.train_test_split`,
you can create your own skorch :class:`~skorch.dataset.Dataset`, and then pass
it to :func:`~skorch.helper.predefined_split`:

.. code:: python

    from sklearn.model_selection import train_test_split
    from skorch.helper import predefined_split
    from skorch.dataset import Dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    valid_ds = Dataset(X_test, y_test)

    net = NeuralNet(
        ...,
        train_split=predefined_split(valid_ds)
    )

    net.fit(X_train, y_train)


What happens when NeuralNet is passed an initialized Pytorch module?
--------------------------------------------------------------------

When :class:`~skorch.net.NeuralNet` is passed an initialized Pytorch module,
skorch will usually leave the module alone. In the following example, the
resulting module will be trained for 20 epochs:

.. code:: python

    class MyModule(nn.Module):
        def __init__(self, hidden=10):
            ...

    module = MyModule()
    net1 = NeuralNet(module, max_epochs=10, ...)
    net1.fit(X, y)

    net2 = NeuralNet(module, max_epochs=10, ...)
    net2.fit(X, y)

When the module is passed to the second :class:`~skorch.net.NeuralNet`, it
will not be re-initialized and will keep its parameters from the first 10
epochs.

When the module parameters are set through keywords arguments,
:class:`~skorch.net.NeuralNet` will re-initialized the module:

.. code:: python

    net = NeuralNet(module, module__hidden=10, ...)
    net.fit(X, y)

Although it is possible to pass an initialized Pytorch module to
:class:`~skorch.net.NeuralNet`, it is recommended to pass the module class
instead:

.. code:: python

    net = NeuralNet(MyModule, ...)
    net.fit(X, y)

In this case, :func:`~skorch.net.NeuralNet.fit` will always re-initialize
the model and :func:`~skorch.net.NeuralNet.partial_fit` won't after the
network is initialized once.

.. _faq_how_do_i_use_a_pytorch_dataset_with_skorch:

How do I use a PyTorch Dataset with skorch?
-------------------------------------------

skorch supports PyTorch's :class:`~torch.utils.data.Dataset` as arguments to
:func:`~skorch.net.NeuralNet.fit` or 
:func:`~skorch.net.NeuralNet.partial_fit`. We create a dataset by 
subclassing PyTorch's :class:`~torch.utils.data.Dataset`:

.. code:: python

    import torch.utils.data
 
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.X = torch.randn(128, 10)
            self.Y = torch.randn(128, 10)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

        def __len__(self):
            return 128

skorch expects the output of ``__getitem__`` to be a tuple of two values. 
The ``RandomDataset`` can be passed directly to 
:func:`~skorch.net.NeuralNet.fit`:

.. code:: python

    from skorch import NeuralNet
    import torch.nn as nn

    train_ds = RandomDataset()

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 10)
            
        def forward(self, X):
            return self.layer(X)

    net = NeuralNet(MyModule, criterion=torch.nn.MSELoss)
    net.fit(train_ds)


How can I deal with multiple return values from forward?
--------------------------------------------------------

skorch supports modules that return multiple values. To do this,
simply return a tuple of all values that you want to return from the
``forward`` method. However, this tuple will also be passed to the
criterion. If the criterion cannot deal with multiple values, this
will result in an error.

To remedy this, you need to either implement your own criterion that
can deal with the output or you need to override
:func:`~skorch.net.NeuralNet.get_loss` and handle the unpacking of the
tuple.

To inspect all output values, you can use either the
:func:`~skorch.net.NeuralNet.forward` method (eager) or the
:func:`~skorch.net.NeuralNet.forward_iter` method (lazy).

For an example of how this works, have a look at this `notebook
<https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Multiple-return-values-from-forward>`_.

How can I perform gradient accumulation with skorch?
----------------------------------------------------

There is no direct option to turn on gradient accumulation (at least
for now). However, with a few modifications, you can implement
gradient accumulation yourself:


.. code:: python

    ACC_STEPS = 2  # number of steps to accumulate before updating weights

    class GradAccNet(NeuralNetClassifier):
        """Net that accumulates gradients"""
        def __init__(self, *args, acc_steps=ACC_STEPS, **kwargs):
            super().__init__(*args, **kwargs)
            self.acc_steps = acc_steps

        def get_loss(self, *args, **kwargs):
            loss = super().get_loss(*args, **kwargs)
            return loss / self.acc_steps  # normalize loss

        def train_step(self, Xi, yi, **fit_params):
            """Perform gradient accumulation

            Only optimize every nth batch.

            """
            # note that n_train_batches starts at 1 for each epoch
            n_train_batches = len(self.history[-1, 'batches'])
            step = self.train_step_single(Xi, yi, **fit_params)

            if n_train_batches % self.acc_steps == 0:
                self.optimizer_.step()
                self.optimizer_.zero_grad()
            return step

This is not a complete recipe. For example, if you optimize every 2nd
step, and the number of training batches is uneven, you should make
sure that there is an optimization step after the last batch of each
epoch. However, this example can serve as a starting point to
implement your own version gradient accumulation.

How can I dynamically set the input size of the PyTorch module based on the data?
---------------------------------------------------------------------------------

Typically, it's up to the user to determine the shape of the input
data when defining the PyTorch module. This can sometimes be
inconvenient, e.g. when the shape is only known at runtime. E.g., when
using :class:`sklearn.feature_selection.VarianceThreshold`, you cannot
know the number of features in advance. The best solution would be to
set the input size dynamically.

In most circumstances, this can be achieved with a few lines of code
in skorch. Here is an example:

.. code:: python

    class InputShapeSetter(skorch.callbacks.Callback):
        def on_train_begin(self, net, X, y):
            net.set_params(module__input_dim=X.shape[1])


    net = skorch.NeuralNetClassifier(
        ClassifierModule,
        callbacks=[InputShapeSetter()],
    )

This assumes that your module accepts an argument called
``input_units``, which determines the number of units of the input
layer, and that the number of features can be determined by
``X.shape[1]``. If those assumptions are not true for your case,
adjust the code accordingly. A fully working example can be found
on `stackoverflow <https://stackoverflow.com/a/60170023/1643939>`_.

How do I implement a score method on the net that returns the loss?
-------------------------------------------------------------------

Sometimes, it is useful to be able to compute the loss of a net from within
``skorch`` (e.g. when a net is part of an ``sklearn`` pipeline). The function
:func:`skorch.scoring.loss_scoring` achieves this. Two examples are provided
below. The first demonstrates how to use :func:`skorch.scoring.loss_scoring` as
a function on a trained ``net`` object.

.. code:: python

    from skorch.scoring import loss_scoring

    X = np.random.randn(250, 25).astype('float32')
    y = (X.dot(np.ones(25)) > 0).astype(int)

    module = nn.Sequential(
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 2),
        nn.Softmax(dim=1)
    )
    net = skorch.NeuralNetClassifier(module).fit(X, y)
    print(loss_scoring(net, X, y))

The second example shows how to sub-class :class:`skorch.classifier.NeuralNetClassifier` to
implement a ``score`` method. In this example, the ``score`` method returns the
**negative** of the loss value, because we want
:class:`sklearn.model_selection.GridSearchCV` to return the run with **least**
loss and :class:`sklearn.model_selection.GridSearchCV` searches for the run with
the **greatest** score.

.. code:: python

    class ScoredNet(skorch.NeuralNetClassifier):
        def score(self, X, y=None):
            loss_value = loss_scoring(self, X, y)
            return -loss_value
    
    net = ScoredNet(module)
    grid_searcher = GridSearchCV(
        net, {'lr': [1e-2, 1e-3], 'batch_size': [8, 16]},
    )
    grid_searcher.fit(X, y)
    best_net = grid_searcher.best_estimator_
    print(best_net.score(X, y))
