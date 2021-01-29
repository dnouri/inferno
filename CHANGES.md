# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `SacredLogger` callback for logging to Sacred (#725)
- CLI helper function now also supports normal (i.e. non-skorch) sklearn estimators
- Disabling all callbacks is now supported (which allows reducing overhead, which is especially relevant for small models)
- Add an example of using skorch with [optuna](https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/optuna-example.ipynb) (#718)

### Changed

- We no longer pass the `epoch` parameter to LR schedulers, since that parameter has been deprecated. We now rely on the scheduler to keep track of the epoch itself.
- Changed implementation of `net.history` access to make it faster; this should result in a nice speedup when dealing with very small model/data but otherwise not have any noticeable effects; if you encounter bugs, though, please create an issue

### Fixed

## [0.9.0] - 2020-08-30

### Added

- Added the `event_name` argument for `LRScheduler` for optional recording of LR changes inside `net.history`. NOTE: Supported only in Pytorch>=1.4
- Make it easier to add custom modules or optimizers to a neural net class by automatically registering them where necessary and by making them available to set_params
- Added the `step_every` argument for `LRScheduler` to set whether the scheduler step should be taken on every epoch or on every batch.
- Added the `scoring` module with `loss_scoring` function, which computes the net's loss (using `get_loss`) on provided input data.
- Added a parameter `predict_nonlinearity` to `NeuralNet` which allows users to control the nonlinearity to be applied to the module output when calling `predict` and `predict_proba` (#637, #661)
- Added the possibility to save the criterion with `save_params` and with checkpoint callbacks
- Added the possibility to save custom modules with `save_params` and with checkpoint callbacks

### Changed

- Removed support for schedulers with a `batch_step()` method in `LRScheduler`. 
- Raise `FutureWarning` in `CVSplit` when `random_state` is not used. Will raise an exception in a future (#620)
- The behavior of method `net.get_params` changed to make it more consistent with sklearn: it will no longer return "learned" attributes like `module_`; therefore, functions like `sklearn.base.clone`, when called with a fitted net, will no longer return a fitted net but instead an uninitialized net; if you want a copy of a fitted net, use `copy.deepcopy` instead;`net.get_params` is used under the hood by many sklearn functions and classes, such as `GridSearchCV`, whose behavior may thus be affected by the change. (#521, #527)
- Raise `FutureWarning` when using `CyclicLR` scheduler, because the default behavior has changed from taking a step every batch to taking a step every epoch. (#626)
- Set train/validation on criterion if it's a PyTorch module (#621)
- Don't pass `y=None` to `NeuralNet.train_split` to enable the direct use of split functions without positional `y` in their signatures. This is useful when working with unsupervised data (#605).
- `to_numpy` is now able to unpack dicts and lists/tuples (#657, #658)
- When using `CrossEntropyLoss`, softmax is now automatically applied to the output when calling `predict` or `predict_proba`

### Fixed

- Fixed a bug where `CyclicLR` scheduler would update during both training and validation rather than just during training.
- Fixed a bug introduced by moving the `optimizer.zero_grad()` call outside of the train step function, making it incompatible with LBFGS and other optimizers that call the train step several times per batch (#636)
- Fixed pickling of the `ProgressBar` callback (#656)

## [0.8.0] - 2020-04-11

### Added

- Added `NeptuneLogger` callback for logging experiment metadata to neptune.ai (#586)
- Add `DataFrameTransformer`, an sklearn compatible transformer that helps working with pandas DataFrames by transforming the DataFrame into a representation that works well with neural networks (#507)
- Added `WandbLogger` callback for logging to Weights & Biases (#607)
- Added `None` option to `device` which leaves the device(s) unmodified (#600)
- Add `PassthroughScoring`, a scoring callback that just calculates the average score of a metric determined at batch level and then writes it to the epoch level (#595)

### Changed

- When using caching in scoring callbacks, no longer uselessly iterate over the data; this can save time if iteration is slow (#552, #557)
- Cleaned up duplicate code in the `fit_loop` (#564)

### Future Changes

- WARNING: In release 0.10.0 of skorch, Python 3.5 support will be officially dropped (#634)

### Fixed

- Make skorch compatible with sklearn 0.22 (#571, #573, #575)
- Fixed a bug that could occur when a new "settable" (via `set_params`) attribute was added to `NeuralNet` whose name starts the same as an existing attribute's name (#590)

## [0.7.0] - 2019-11-29

### Added

- More careful check for wrong parameter names being passed to `NeuralNet` (#500)
- More helpful error messages when trying to predict using an uninitialized model
- Add `TensorBoard` callback for automatic logging to tensorboard
- Make `NeuralNetBinaryClassifier` work with `sklearn.calibration.CalibratedClassifierCV`
- Improve `NeuralNetBinaryClassifier` compatibility with certain sklearn metrics (#515)
- `NeuralNetBinaryClassifier` automatically squeezes module output if necessary (#515)
- `NeuralNetClassifier` now has a `classes_` attribute after fit is called, which is inferred from y by default (#465, #486)
- `NeuralNet.load_params` with a checkpoint now initializes when needed (#497)

### Changed

- Improve numerical stability when using `NLLLoss` in `NeuralNetClassifer` (#491)
- Refactor code to make gradient accumulation easier to implement (#506)
- `NeuralNetBinaryClassifier.predict_proba` now returns a 2-dim array; to access the "old" `y_proba`, take `y_proba[:, 1]` (#515)
- `net.history` is now a property that accesses `net.history_`, which stores the `History` object (#527)
- Remove deprecated `skorch.callbacks.CyclicLR`, use `torch.optim.lr_scheduler.CyclicLR` instead

### Future Changes

- WARNING: In a future release, the behavior of method `net.get_params` will change to make it more consistent with sklearn: it will no longer return "learned" attributes like `module_`. Therefore, functions like `sklearn.base.clone`, when called with a fitted net, will no longer return a fitted net but instead an uninitialized net. If you want a copy of a fitted net, use `copy.deepcopy` instead. Note that `net.get_params` is used under the hood by many sklearn functions and classes, such as `GridSearchCV`, whose behavior may thus be affected by the change. (#521, #527)

### Fixed

- Fixed a bug that caused `LoadInitState` not to work with `TrainEndCheckpoint` (#528)
- Fixed `NeuralNetBinaryClassifier` wrongly squeezing the batch dimension when using `batch_size = 1` (#558)


## [0.6.0] - 2019-07-19

### Added

- Adds FAQ entry regarding the initialization behavior of `NeuralNet` when passed instantiated models. (#409)
- Added CUDA pickle test including an artifact that supports testing on CUDA-less CI machines
- Adds `train_batch_count` and `valid_batch_count` to history in training loop. (#445)
- Adds score method for NeuralNetClassifier, NeuralNetBinaryClassifier, and NeuralNetRegressor (#469)
- Wrapper class for torch Datasets to make them work with some sklearn features (e.g. grid search). (#443)

### Changed

- Repository moved to https://github.com/skorch-dev/skorch/, please change your git remotes
- Treat cuda dependent attributes as prefix to cover values set using `set_params` since
  previously `"criterion_"` would not match `net.criterion__weight` as set by
  `net.set_params(criterion__weight=w)`
- skorch pickle format changed in order to improve CUDA compatibility, if you have pickled models, please re-pickle them to be able to load them in the future
- `net.criterion_` and its parameters are now moved to target device when using criteria that inherit from `torch.nn.Module`. Previously the user had to make sure that parameters such as class weight are on the compute device
- skorch now assumes PyTorch >= 1.1.0. This mainly affects learning rate schedulers, whose inner workings have been changed with version 1.1.0. This update will also invalidate pickled skorch models after a change introduced in PyTorch optimizers.

### Fixed

- Include requirements in MANIFEST.in
- Add `criterion_` to `NeuralNet.cuda_dependent_attributes_` to avoid issues with criterion
  weight tensors from, e.g., `NLLLoss` (#426)
- `TrainEndCheckpoint` can be cloned by `sklearn.base.clone`. (#459)


## [0.5.0] - 2018-12-13

### Added

- [Basic usage notebook][1810251445] now runs on Google Colab
- [Advanced usage notebook][1810261633] now runs on Google Colab
- [MNIST with scikit-learn and skorch][1811011230] now runs on Google Colab
- Better user-facing messages when module or optimizer are re-initialized
- Added an experimental API (`net._register_virtual_param`) to register "virtual"
  parameters on the network with custom setter functions. (#369)
- Setting parameters `lr`, `momentum`, `optimizer__lr`, etc. no longer resets
  the optmizer. As of now you can do `net.set_params(lr=0.03)` or
  `net.set_params(optimizer__param_group__0__momentum=0.86)` without triggering
  a re-initialization of the optimizer (#369)
- Support for scipy sparse CSR matrices as input (as, e.g., returned by sklearn's
  `CountVectorizer`); note that they are cast to dense matrices during batching
- Helper functions to build command line interfaces with almost no
  boilerplate, [example][1811191713] that shows usage

[1810251445]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb
[1810261633]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb
[1811011230]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb
[1811191713]: https://github.com/skorch-dev/skorch/tree/master/examples/cli

### Changed

- Reduce overhead of `BatchScoring` when using `train_loss_score` or `valid_loss_score` by skipping superfluous inference step (#381)
- The `on_grad_computed` callback function will yield an iterable for `named_parameters` only when it is used to reduce the run-time overhead of the call (#379)
- Default `fn_prefix` in `TrainEndCheckpoint` is now `train_end_` (#391)
- Issues a warning when `Checkpoints`'s `monitor` parameter is set to `monitor` and the history contains `<monitor>_best`. (#399)

### Fixed

- Re-initialize optimizer when `set_params` is called with `lr` argument (#372)
- Copying a `SliceDict` now returns a `SliceDict` instead of a `dict` (#388)
- Calling `==` on `SliceDict`s now works as expected when values are numpy arrays and torch tensors


## [0.4.0] - 2018-10-24

### Added

- Support for PyTorch 0.4.1
- There is no need to explicitly name callbacks anymore (names are assigned automatically, name conflicts are resolved).
- You can now access the training data in the `on_grad_computed` event
- There is a new [image segmentation example][1]
- Easily create toy network instances for quick experiments using [`skorch.toy.make_classifier`][2] and friends
- New [`ParamMapper`][3] callback to modify/freeze/unfreeze parameters at certain point in time during training:
```python
>>> from sklearn.callbacks import Freezer, Unfreezer
>>> net = Net(module, callbacks=[Freezer('layer*.weight'), Unfreezer('layer*.weight', at=10)])
```
- Refactored `EpochScoring` for easier sub-classing
- `Checkpoint` callback now supports saving the optimizer, this avoids problems with stateful
  optimizers such as `Adam` or `RMSprop` (#360)
- Added `LoadInitState` callback for easy continued training from checkpoints (#360)
- `NeuralNetwork.load_params` now supports loading from `Checkpoint` instances
- Added documentation for [saving and loading][4]

[1]: https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/Nuclei_Image_Segmentation.ipynb
[2]: https://skorch.readthedocs.io/en/latest/toy.html
[3]: https://skorch.readthedocs.io/en/latest/callbacks.html#skorch.callbacks.ParamMapper
[4]: https://skorch.readthedocs.io/en/latest/user/save_load.html

### Changed

- The `ProgressBar` callback now determines the batches per epoch automatically by default (`batches_per_epoch=auto`)
- The `on_grad_computed` event now has access to the current training data batch

### Deprecated

- Deprecated `filtered_optimizer` in favor of `Freezer` callback (#346)
- `NeuralNet.load_params` and `NeuralNet.save_params` deprecate `f` parameter for the sake
  of `f_optimizer`, `f_params` and `f_history` (#360)

### Fixed

- `uses_placeholder_y` should not require existence of `y` field (#311)
- LR scheduler creates `batch_idx` on first run (#314)
- Use `OrderedDict` for callbacks to fix python 3.5 compatibility issues (#331)
- Make `to_tensor` work correctly with `PackedSequence` (#335)
- Rewrite `History` to not use any recursion to avoid memory leaks during exceptions (#312)
- Use `flaky` in some neural network tests to hide platform differences
- Fixes ReduceLROnPlateau when mode == max (#363)
- Fix disconnected weights between net and optimizer after copying the net with `copy.deepcopy` (#318)
- Fix a bug that intefered with loading CUDA models when the model was a CUDA tensor but
  the net was configured to use the CPU (#354, #358)


[Unreleased]: https://github.com/skorch-dev/skorch/compare/v0.9.0...HEAD
[0.4.0]: https://github.com/skorch-dev/skorch/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/skorch-dev/skorch/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/skorch-dev/skorch/compare/v0.5.0...v0.6.0
[0.7.0]: https://github.com/skorch-dev/skorch/compare/v0.6.0...v0.7.0
[0.8.0]: https://github.com/skorch-dev/skorch/compare/v0.7.0...v0.8.0
[0.9.0]: https://github.com/skorch-dev/skorch/compare/v0.8.0...v0.9.0
