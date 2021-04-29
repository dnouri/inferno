"""Tests for net.py

Although NeuralNetClassifier is used in tests, test only functionality
that is general to NeuralNet class.

"""

import copy
from distutils.version import LooseVersion
from functools import partial
import os
from pathlib import Path
import pickle
import re
from unittest.mock import Mock
from unittest.mock import patch
import sys
import time
from contextlib import ExitStack

from flaky import flaky
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from skorch.tests.conftest import INFERENCE_METHODS
from skorch.utils import flatten
from skorch.utils import to_numpy
from skorch.utils import is_torch_data_type


ACCURACY_EXPECTED = 0.65


# pylint: disable=too-many-public-methods
class TestNeuralNet:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope='module')
    def dummy_callback(self):
        from skorch.callbacks import Callback
        cb = Mock(spec=Callback)
        # make dummy behave like an estimator
        cb.get_params.return_value = {}
        cb.set_params = lambda **kwargs: cb
        return cb

    @pytest.fixture(scope='module')
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def dataset_cls(self):
        from skorch.dataset import Dataset
        return Dataset

    @pytest.fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls, dummy_callback):
        return net_cls(
            module_cls,
            callbacks=[('dummy', dummy_callback)],
            max_epochs=10,
            lr=0.1,
        )

    @pytest.fixture(scope='module')
    def pipe(self, net):
        return Pipeline([
            ('scale', StandardScaler()),
            ('net', net),
        ])

    @pytest.fixture(scope='module')
    def net_fit(self, net_cls, module_cls, dummy_callback, data):
        # Careful, don't call additional fits or set_params on this,
        # since that would have side effects on other tests.
        X, y = data

        # We need a new instance of the net and cannot reuse the net
        # fixture, because otherwise fixture net and net_fit refer to
        # the same object; also, we cannot clone(net) because this
        # will result in the dummy_callback not being the mock anymore
        net = net_cls(
            module_cls,
            callbacks=[('dummy', dummy_callback)],
            max_epochs=10,
            lr=0.1,
        )
        return net.fit(X, y)

    @pytest.fixture
    def net_pickleable(self, net_fit):
        """NeuralNet instance that removes callbacks that are not
        pickleable.

        """
        # callback fixture not pickleable, remove it
        callbacks = net_fit.callbacks
        net_fit.callbacks = []
        callbacks_ = net_fit.callbacks_
        # remove mock callback
        net_fit.callbacks_ = [(n, cb) for n, cb in net_fit.callbacks_
                              if not isinstance(cb, Mock)]
        net_clone = copy.deepcopy(net_fit)
        net_fit.callbacks = callbacks
        net_fit.callbacks_ = callbacks_
        return net_clone

    @pytest.mark.parametrize("copy_method", ["pickle", "copy.deepcopy"])
    def test_train_net_after_copy(self, net_cls, module_cls, data,
                                  copy_method):
        # This test comes from [issue #317], and makes sure that models
        # can be trained after copying (which is really pickling).
        #
        # [issue #317]:https://github.com/skorch-dev/skorch/issues/317
        X, y = data
        n1 = net_cls(module_cls)
        n1.partial_fit(X, y, epochs=1)
        if copy_method == "copy.deepcopy":
            n2 = copy.deepcopy(n1)
        elif copy_method == "pickle":
            n2 = pickle.loads(pickle.dumps(n1))
        else:
            raise ValueError

        # Test to make sure the parameters got copied correctly
        close = [torch.allclose(p1, p2)
                 for p1, p2 in zip(n1.module_.parameters(),
                                   n2.module_.parameters())]
        assert all(close)

        # make sure the parameters change
        # at least two epochs to make sure `train_loss` updates after copy
        # (this is a check for the bug in #317, where `train_loss` didn't
        # update at all after copy. This covers that case).
        n2.partial_fit(X, y, epochs=2)
        far = [not torch.allclose(p1, p2)
               for p1, p2 in zip(n1.module_.parameters(),
                                 n2.module_.parameters())]
        assert all(far)

        # Make sure the model is being trained, and the loss actually changes
        # (and hopefully decreases, but no test for that)
        # If copied incorrectly, the optimizer can't see the gradients
        # calculated by loss.backward(), so the loss stays *exactly* the same
        assert n2.history[-1]['train_loss'] != n2.history[-2]['train_loss']

        # Make sure the optimizer params and module params point to the same
        # memory
        for opt_param, param in zip(
                n2.module_.parameters(),
                n2.optimizer_.param_groups[0]['params']):
            assert param is opt_param

    def test_net_init_one_unknown_argument(self, net_cls, module_cls):
        with pytest.raises(TypeError) as e:
            net_cls(module_cls, unknown_arg=123).initialize()

        expected = ("__init__() got unexpected argument(s) unknown_arg. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
        assert e.value.args[0] == expected

    def test_net_init_two_unknown_arguments(self, net_cls, module_cls):
        with pytest.raises(TypeError) as e:
            net_cls(module_cls, lr=0.1, mxa_epochs=5,
                    warm_start=False, bathc_size=20).initialize()

        expected = ("__init__() got unexpected argument(s) "
                    "bathc_size, mxa_epochs. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
        assert e.value.args[0] == expected

    @pytest.mark.parametrize('name, suggestion', [
        ('iterator_train_shuffle', 'iterator_train__shuffle'),
        ('optimizer_momentum', 'optimizer__momentum'),
        ('modulenum_units', 'module__num_units'),
        ('criterionreduce', 'criterion__reduce'),
        ('callbacks_mycb__foo', 'callbacks__mycb__foo'),
    ])
    def test_net_init_missing_dunder_in_prefix_argument(
            self, net_cls, module_cls, name, suggestion):
        # forgot to use double-underscore notation
        with pytest.raises(TypeError) as e:
            net_cls(module_cls, **{name: 123}).initialize()

        tmpl = "Got an unexpected argument {}, did you mean {}?"
        expected = tmpl.format(name, suggestion)
        assert e.value.args[0] == expected

    def test_net_init_missing_dunder_in_2_prefix_arguments(
            self, net_cls, module_cls):
        # forgot to use double-underscore notation in 2 arguments
        with pytest.raises(TypeError) as e:
            net_cls(
                module_cls,
                max_epochs=7,  # correct
                iterator_train_shuffle=True,  # uses _ instead of __
                optimizerlr=0.5,  # missing __
            ).initialize()
        expected = ("Got an unexpected argument iterator_train_shuffle, "
                    "did you mean iterator_train__shuffle?\n"
                    "Got an unexpected argument optimizerlr, "
                    "did you mean optimizer__lr?")
        assert e.value.args[0] == expected

    def test_net_init_missing_dunder_and_unknown(
            self, net_cls, module_cls):
        # unknown argument and forgot to use double-underscore notation
        with pytest.raises(TypeError) as e:
            net_cls(
                module_cls,
                foobar=123,
                iterator_train_shuffle=True,
            ).initialize()
        expected = ("__init__() got unexpected argument(s) foobar. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.\n"
                    "Got an unexpected argument iterator_train_shuffle, "
                    "did you mean iterator_train__shuffle?")
        assert e.value.args[0] == expected

    def test_net_with_new_attribute_with_name_clash(
            self, net_cls, module_cls):
        # This covers a bug that existed when a new "settable"
        # argument was added whose name starts the same as the name
        # for an existing argument
        class MyNet(net_cls):
            # add "optimizer_2" as a valid prefix so that it works
            # with set_params
            prefixes_ = net_cls.prefixes_[:] + ['optimizer_2']

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.optimizer_2 = torch.optim.SGD

        # the following line used to raise this error: "TypeError: Got
        # an unexpected argument optimizer_2__lr, did you mean
        # optimizer__2__lr?" because it was erronously assumed that
        # "optimizer_2__lr" should be dispatched to "optimizer", not
        # "optimizer_2".
        MyNet(module_cls, optimizer_2__lr=0.123)  # should not raise

    def test_net_init_with_iterator_valid_shuffle_true(
            self, net_cls, module_cls, recwarn):
        # If a user sets iterator_valid__shuffle=True, they might be
        # in for a surprise, since predict et al. will result in
        # shuffled predictions. It is best to warn about this, since
        # most of the times, this is not what users actually want.
        expected = (
            "You set iterator_valid__shuffle=True; this is most likely not what you want "
            "because the values returned by predict and predict_proba will be shuffled.")

        # no warning expected here
        net_cls(module_cls, iterator_valid__shuffle=False)
        assert not recwarn.list

        # warning expected here
        with pytest.warns(UserWarning, match=expected):
            net_cls(module_cls, iterator_valid__shuffle=True).initialize()

    def test_fit(self, net_fit):
        # fitting does not raise anything
        pass

    @pytest.mark.parametrize('method', INFERENCE_METHODS)
    def test_not_fitted_raises(self, net_cls, module_cls, data, method):
        from skorch.exceptions import NotInitializedError
        net = net_cls(module_cls)
        X = data[0]
        with pytest.raises(NotInitializedError) as exc:
            # we call `list` because `forward_iter` is lazy
            list(getattr(net, method)(X))

        msg = ("This NeuralNetClassifier instance is not initialized yet. "
               "Call 'initialize' or 'fit' with appropriate arguments "
               "before using this method.")
        assert exc.value.args[0] == msg

    def test_not_fitted_other_attributes(self, module_cls):
        # pass attributes to check for explicitly
        with patch('skorch.net.check_is_fitted') as check:
            from skorch import NeuralNetClassifier

            net = NeuralNetClassifier(module_cls)
            attributes = ['foo', 'bar_']

            net.check_is_fitted(attributes=attributes)
            args = check.call_args_list[0][0][1]
            assert args == attributes

    def test_net_learns(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(
            module_cls,
            max_epochs=10,
            lr=0.1,
        )
        net.fit(X, y)
        y_pred = net.predict(X)
        assert accuracy_score(y, y_pred) > ACCURACY_EXPECTED

    def test_forward(self, net_fit, data):
        X = data[0]
        n = len(X)
        y_forward = net_fit.forward(X)

        assert is_torch_data_type(y_forward)
        # Expecting (number of samples, number of output units)
        assert y_forward.shape == (n, 2)

        y_proba = net_fit.predict_proba(X)
        assert np.allclose(to_numpy(y_forward), y_proba)

    def test_forward_device_cpu(self, net_fit, data):
        X = data[0]

        # CPU by default
        y_forward = net_fit.forward(X)
        assert isinstance(X, np.ndarray)
        assert not y_forward.is_cuda

        y_forward = net_fit.forward(X, device='cpu')
        assert isinstance(X, np.ndarray)
        assert not y_forward.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_forward_device_gpu(self, net_fit, data):
        X = data[0]
        y_forward = net_fit.forward(X, device='cuda:0')
        assert isinstance(X, np.ndarray)
        assert y_forward.is_cuda

    def test_dropout(self, net_fit, data):
        # Note: does not test that dropout is really active during
        # training.
        X = data[0]

        # check that dropout not active by default
        y_proba = to_numpy(net_fit.forward(X))
        y_proba2 = to_numpy(net_fit.forward(X))
        assert np.allclose(y_proba, y_proba2, rtol=1e-7)

        # check that dropout can be activated
        y_proba = to_numpy(net_fit.forward(X, training=True))
        y_proba2 = to_numpy(net_fit.forward(X, training=True))
        assert not np.allclose(y_proba, y_proba2, rtol=1e-7)

    def test_pickle_save_load(self, net_pickleable, data, tmpdir):
        X, y = data
        score_before = accuracy_score(y, net_pickleable.predict(X))

        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        with open(str(p), 'wb') as f:
            pickle.dump(net_pickleable, f)
        del net_pickleable
        with open(str(p), 'rb') as f:
            net_new = pickle.load(f)

        score_after = accuracy_score(y, net_new.predict(X))
        assert np.isclose(score_after, score_before)

    def train_picklable_cuda_net(self, net_pickleable, data):
        X, y = data
        w = torch.FloatTensor([1.] * int(y.max() + 1)).to('cuda')

        # Use stateful optimizer (CUDA variables in state) and
        # a CUDA parametrized criterion along with a CUDA net.
        net_pickleable.set_params(
            device='cuda',
            criterion__weight=w,
            optimizer=torch.optim.Adam,
        )
        net_pickleable.fit(X, y)

        return net_pickleable

    @pytest.fixture
    def pickled_cuda_net_path(self, net_pickleable, data):
        path = os.path.join('skorch', 'tests', 'net_cuda.pkl')

        # Assume that a previous run on a CUDA-capable device
        # created `net_cuda.pkl`.
        if not torch.cuda.is_available():
            assert os.path.exists(path)
            return path

        net_pickleable = self.train_picklable_cuda_net(net_pickleable, data)

        with open(path, 'wb') as f:
            pickle.dump(net_pickleable, f)
        return path

    @pytest.mark.parametrize('cuda_available', {False, torch.cuda.is_available()})
    def test_pickle_load(self, cuda_available, pickled_cuda_net_path):
        with patch('torch.cuda.is_available', lambda *_: cuda_available):
            with open(pickled_cuda_net_path, 'rb') as f:
                pickle.load(f)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_device_torch_device(self, net_cls, module_cls, device):
        # Check if native torch.device works as well.
        if device.startswith('cuda') and not torch.cuda.is_available():
            pytest.skip()
        net = net_cls(module=module_cls, device=torch.device(device))
        net = net.initialize()
        assert net.module_.sequential[0].weight.device.type.startswith(device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    @pytest.mark.parametrize(
        'save_dev, cuda_available, load_dev, expect_warning',
        [
            ('cuda', False, 'cpu', True),
            ('cuda', True, 'cuda', False),
            ('cpu', True, 'cpu', False),
            ('cpu', False, 'cpu', False),
        ])
    def test_pickle_save_and_load_mixed_devices(
            self,
            net_cls,
            module_cls,
            tmpdir,
            save_dev,
            cuda_available,
            load_dev,
            expect_warning,
    ):
        from skorch.exceptions import DeviceWarning
        net = net_cls(module=module_cls, device=save_dev).initialize()

        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        with open(str(p), 'wb') as f:
            pickle.dump(net, f)
        del net

        with patch('torch.cuda.is_available', lambda *_: cuda_available):
            with open(str(p), 'rb') as f:
                expected_warning = DeviceWarning if expect_warning else None
                with pytest.warns(expected_warning) as w:
                    m = pickle.load(f)

        assert torch.device(m.device) == torch.device(load_dev)

        if expect_warning:
            # We should have captured two warnings:
            # 1. one for the failed load
            # 2. for switching devices on the net instance
            assert len(w.list) == 2
            assert w.list[0].message.args[0] == (
                'Requested to load data to CUDA but no CUDA devices '
                'are available. Loading on device "cpu" instead.')
            assert w.list[1].message.args[0] == (
                'Setting self.device = {} since the requested device ({}) '
                'is not available.'.format(load_dev, save_dev))

    def test_pickle_save_and_load_uninitialized(
            self, net_cls, module_cls, tmpdir):
        net = net_cls(module_cls)
        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        with open(str(p), 'wb') as f:
            # does not raise
            pickle.dump(net, f)
        with open(str(p), 'rb') as f:
            pickle.load(f)

    def test_save_params_invalid_argument_name_raises(self, net_fit):
        msg = ("save_params got an unexpected argument 'foobar', "
               "did you mean 'f_foobar'?")
        with pytest.raises(TypeError, match=msg):
            net_fit.save_params(foobar='some-file.pt')

    def test_load_params_invalid_argument_name_raises(self, net_fit):
        msg = ("load_params got an unexpected argument 'foobar', "
               "did you mean 'f_foobar'?")
        with pytest.raises(TypeError, match=msg):
            net_fit.load_params(foobar='some-file.pt')

    def test_save_params_with_f_params_and_f_module_raises(self, net_fit):
        msg = "save_params called with both f_params and f_module, please choose one"
        with pytest.raises(TypeError, match=msg):
            net_fit.save_params(f_module='weights.pt', f_params='params.pt')

    def test_load_params_with_f_params_and_f_module_raises(self, net_fit):
        msg = "load_params called with both f_params and f_module, please choose one"
        with pytest.raises(TypeError, match=msg):
            net_fit.load_params(f_module='weights.pt', f_params='params.pt')

    def test_save_params_no_state_dict_raises(self, net_fit):
        msg = ("You are trying to save 'f_max_epochs' but for that to work, the net "
               "needs to have an attribute called 'net.max_epochs_' that is a PyTorch "
               "Module or Optimizer; make sure that it exists and check for typos.")
        with pytest.raises(AttributeError, match=msg):
            net_fit.save_params(f_max_epochs='some-file.pt')

    def test_load_params_no_state_dict_raises(self, net_fit):
        msg = ("You are trying to load 'f_max_epochs' but for that to work, the net "
               "needs to have an attribute called 'net.max_epochs_' that is a PyTorch "
               "Module or Optimizer; make sure that it exists and check for typos.")
        with pytest.raises(AttributeError, match=msg):
            net_fit.load_params(f_max_epochs='some-file.pt')

    def test_save_params_unknown_attribute_raises(self, net_fit):
        msg = ("You are trying to save 'f_unknown' but for that to work, the net "
               "needs to have an attribute called 'net.unknown_' that is a PyTorch "
               "Module or Optimizer; make sure that it exists and check for typos.")
        with pytest.raises(AttributeError, match=msg):
            net_fit.save_params(f_unknown='some-file.pt')

    def test_load_params_unknown_attribute_raises(self, net_fit):
        msg = ("You are trying to load 'f_unknown' but for that to work, the net "
               "needs to have an attribute called 'net.unknown_' that is a PyTorch "
               "Module or Optimizer; make sure that it exists and check for typos.")
        with pytest.raises(AttributeError, match=msg):
            net_fit.load_params(f_unknown='some-file.pt')

    def test_save_load_state_dict_file(
            self, net_cls, module_cls, net_fit, data, tmpdir):
        net = net_cls(module_cls).initialize()
        X, y = data

        score_before = accuracy_score(y, net_fit.predict(X))
        score_untrained = accuracy_score(y, net.predict(X))
        assert not np.isclose(score_before, score_untrained)

        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        with open(str(p), 'wb') as f:
            net_fit.save_params(f_params=f)
        del net_fit
        with open(str(p), 'rb') as f:
            net.load_params(f_params=f)

        score_after = accuracy_score(y, net.predict(X))
        assert np.isclose(score_after, score_before)

    def test_save_load_state_dict_str(
            self, net_cls, module_cls, net_fit, data, tmpdir):
        net = net_cls(module_cls).initialize()
        X, y = data

        score_before = accuracy_score(y, net_fit.predict(X))
        score_untrained = accuracy_score(y, net.predict(X))
        assert not np.isclose(score_before, score_untrained)

        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        net_fit.save_params(f_params=str(p))
        del net_fit
        net.load_params(f_params=str(p))

        score_after = accuracy_score(y, net.predict(X))
        assert np.isclose(score_after, score_before)

    @pytest.fixture(scope='module')
    def net_fit_adam(self, net_cls, module_cls, data):
        net = net_cls(
            module_cls, max_epochs=2, lr=0.1,
            optimizer=torch.optim.Adam)
        net.fit(*data)
        return net

    @pytest.fixture(scope='module')
    def criterion_with_params_cls(self):
        class MyCriterion(nn.Module):
            """Criterion with learnable parameters"""
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 1)

            def forward(self, y_pred, y_true):
                return ((self.lin(y_pred) - y_true.float()) ** 2).sum()
        return MyCriterion

    @pytest.fixture
    def net_fit_criterion(self, net_cls, module_cls, criterion_with_params_cls, data):
        """Replace criterion by a module so that it has learnt parameters"""
        net = net_cls(
            module_cls,
            criterion=criterion_with_params_cls,
            max_epochs=2,
            lr=0.1,
            optimizer=torch.optim.Adam,
        )
        net.fit(*data)
        return net

    def test_save_load_state_dict_file_with_history_optimizer_criterion(
            self, net_cls, module_cls, criterion_with_params_cls, net_fit_criterion, tmpdir):

        skorch_tmpdir = tmpdir.mkdir('skorch')
        p = skorch_tmpdir.join('testmodel.pkl')
        o = skorch_tmpdir.join('optimizer.pkl')
        c = skorch_tmpdir.join('criterion.pkl')
        h = skorch_tmpdir.join('history.json')

        with ExitStack() as stack:
            p_fp = stack.enter_context(open(str(p), 'wb'))
            o_fp = stack.enter_context(open(str(o), 'wb'))
            c_fp = stack.enter_context(open(str(c), 'wb'))
            h_fp = stack.enter_context(open(str(h), 'w'))
            net_fit_criterion.save_params(
                f_params=p_fp, f_optimizer=o_fp, f_criterion=c_fp, f_history=h_fp)

            # 'step' is state from the Adam optimizer
            orig_steps = [v['step'] for v in
                          net_fit_criterion.optimizer_.state_dict()['state'].values()]
            orig_loss = np.array(net_fit_criterion.history[:, 'train_loss'])
            orig_criterion_weight = dict(
                net_fit_criterion.criterion_.named_parameters())['lin.weight']
            del net_fit_criterion

        with ExitStack() as stack:
            p_fp = stack.enter_context(open(str(p), 'rb'))
            o_fp = stack.enter_context(open(str(o), 'rb'))
            c_fp = stack.enter_context(open(str(c), 'rb'))
            h_fp = stack.enter_context(open(str(h), 'r'))
            new_net = net_cls(
                module_cls,
                criterion=criterion_with_params_cls,
                optimizer=torch.optim.Adam,
            ).initialize()
            new_net.load_params(
                f_params=p_fp, f_optimizer=o_fp, f_criterion=c_fp, f_history=h_fp)

            new_steps = [v['step'] for v in
                         new_net.optimizer_.state_dict()['state'].values()]
            new_loss = np.array(new_net.history[:, 'train_loss'])

            assert np.allclose(orig_loss, new_loss)
            assert orig_steps == new_steps
            new_criterion_weight = dict(new_net.criterion_.named_parameters())[
                'lin.weight']
            assert (orig_criterion_weight == new_criterion_weight).all()

    def test_save_load_state_dict_str_with_history_optimizer(
            self, net_cls, module_cls, net_fit_adam, tmpdir):

        skorch_tmpdir = tmpdir.mkdir('skorch')
        p = str(skorch_tmpdir.join('testmodel.pkl'))
        o = str(skorch_tmpdir.join('optimizer.pkl'))
        h = str(skorch_tmpdir.join('history.json'))

        net_fit_adam.save_params(f_params=p, f_optimizer=o, f_history=h)

        # 'step' is state from the Adam optimizer
        orig_steps = [v['step'] for v in
                      net_fit_adam.optimizer_.state_dict()['state'].values()]
        orig_loss = np.array(net_fit_adam.history[:, 'train_loss'])
        del net_fit_adam

        new_net = net_cls(
            module_cls, optimizer=torch.optim.Adam).initialize()
        new_net.load_params(f_params=p, f_optimizer=o, f_history=h)

        new_steps = [v['step'] for v in
                     new_net.optimizer_.state_dict()['state'].values()]
        new_loss = np.array(new_net.history[:, 'train_loss'])

        assert np.allclose(orig_loss, new_loss)
        assert orig_steps == new_steps

    @pytest.mark.parametrize("explicit_init", [True, False])
    def test_save_and_load_from_checkpoint(
            self, net_cls, module_cls, data, checkpoint_cls, tmpdir,
            explicit_init):

        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join('params.pt')
        f_optimizer = skorch_dir.join('optimizer.pt')
        f_criterion = skorch_dir.join('criterion.pt')
        f_history = skorch_dir.join('history.json')

        cp = checkpoint_cls(
            monitor=None,
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_criterion=str(f_criterion),
            f_history=str(f_history))
        net = net_cls(
            module_cls, max_epochs=4, lr=0.1,
            optimizer=torch.optim.Adam, callbacks=[cp])
        net.fit(*data)
        del net

        assert f_params.exists()
        assert f_optimizer.exists()
        assert f_criterion.exists()
        assert f_history.exists()

        new_net = net_cls(
            module_cls, max_epochs=4, lr=0.1,
            optimizer=torch.optim.Adam, callbacks=[cp])
        if explicit_init:
            new_net.initialize()
        new_net.load_params(checkpoint=cp)

        assert len(new_net.history) == 4

        new_net.partial_fit(*data)

        # fit ran twice for a total of 8 epochs
        assert len(new_net.history) == 8

    def test_checkpoint_with_prefix_and_dirname(
            self, net_cls, module_cls, data, checkpoint_cls, tmpdir):
        exp_dir = tmpdir.mkdir('skorch')
        exp_basedir = exp_dir.join('exp1')

        cp = checkpoint_cls(
            monitor=None, fn_prefix='unet_', dirname=str(exp_basedir))
        net = net_cls(
            module_cls, max_epochs=4, lr=0.1,
            optimizer=torch.optim.Adam, callbacks=[cp])
        net.fit(*data)

        assert exp_basedir.join('unet_params.pt').exists()
        assert exp_basedir.join('unet_optimizer.pt').exists()
        assert exp_basedir.join('unet_history.json').exists()

    def test_save_and_load_from_checkpoint_formatting(
            self, net_cls, module_cls, data, checkpoint_cls, tmpdir):

        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True)

        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join(
            'model_epoch_{last_epoch[epoch]}.pt')
        f_optimizer = skorch_dir.join(
            'optimizer_epoch_{last_epoch[epoch]}.pt')
        f_criterion = skorch_dir.join(
            'criterion_epoch_{last_epoch[epoch]}.pt')
        f_history = skorch_dir.join(
            'history.json')

        cp = checkpoint_cls(
            monitor='epoch_3_scorer',
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_criterion=str(f_criterion),
            f_history=str(f_history))

        net = net_cls(
            module_cls, max_epochs=5, lr=0.1,
            optimizer=torch.optim.Adam, callbacks=[
                ('my_score', scoring), cp
            ])
        net.fit(*data)
        del net

        assert skorch_dir.join('model_epoch_3.pt').exists()
        assert skorch_dir.join('optimizer_epoch_3.pt').exists()
        assert skorch_dir.join('criterion_epoch_3.pt').exists()
        assert skorch_dir.join('history.json').exists()

        new_net = net_cls(
            module_cls, max_epochs=5, lr=0.1,
            optimizer=torch.optim.Adam, callbacks=[
                ('my_score', scoring), cp
            ])
        new_net.load_params(checkpoint=cp)

        # original run saved checkpoint at epoch 3
        assert len(new_net.history) == 3

        new_net.partial_fit(*data)

        # training continued from the best epoch of the first run,
        # the best epoch in the first run happened at epoch 3,
        # the second ran for 5 epochs, so the final history of the new
        # net is 3+5 = 7
        assert len(new_net.history) == 8
        assert new_net.history[:, 'event_cp'] == [
            False, False, True, False, False, False, False, False]

    def test_save_params_not_init_optimizer(
            self, net_cls, module_cls, tmpdir):
        from skorch.exceptions import NotInitializedError

        net = net_cls(module_cls)._initialize_module()
        skorch_tmpdir = tmpdir.mkdir('skorch')
        p = skorch_tmpdir.join('testmodel.pkl')
        o = skorch_tmpdir.join('optimizer.pkl')

        with pytest.raises(NotInitializedError) as exc:
            net.save_params(f_params=str(p), f_optimizer=o)
        expected = ("Cannot save state of an un-initialized model. "
                    "Please initialize first by calling .initialize() "
                    "or by fitting the model with .fit(...).")
        assert exc.value.args[0] == expected

    def test_load_params_not_init_optimizer(
            self, net_cls, module_cls, tmpdir):
        from skorch.exceptions import NotInitializedError

        net = net_cls(module_cls).initialize()
        skorch_tmpdir = tmpdir.mkdir('skorch')
        p = skorch_tmpdir.join('testmodel.pkl')
        net.save_params(f_params=str(p))

        net = net_cls(module_cls)  # not initialized
        o = skorch_tmpdir.join('optimizer.pkl')
        with pytest.raises(NotInitializedError) as exc:
            net.load_params(f_optimizer=str(o))
        expected = ("Cannot load state of an un-initialized model. "
                    "Please initialize first by calling .initialize() "
                    "or by fitting the model with .fit(...).")
        assert exc.value.args[0] == expected

    def test_save_state_dict_not_init(
            self, net_cls, module_cls, tmpdir):
        from skorch.exceptions import NotInitializedError

        net = net_cls(module_cls)
        p = tmpdir.mkdir('skorch').join('testmodel.pkl')

        with pytest.raises(NotInitializedError) as exc:
            net.save_params(f_params=str(p))
        expected = ("Cannot save state of an un-initialized model. "
                    "Please initialize first by calling .initialize() "
                    "or by fitting the model with .fit(...).")
        assert exc.value.args[0] == expected

    def test_load_state_dict_not_init(
            self, net_cls, module_cls, tmpdir):
        from skorch.exceptions import NotInitializedError

        net = net_cls(module_cls)
        p = tmpdir.mkdir('skorch').join('testmodel.pkl')

        with pytest.raises(NotInitializedError) as exc:
            net.load_params(f_params=str(p))
        expected = ("Cannot load state of an un-initialized model. "
                    "Please initialize first by calling .initialize() "
                    "or by fitting the model with .fit(...).")
        assert exc.value.args[0] == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_save_load_state_cuda_intercompatibility(
            self, net_cls, module_cls, tmpdir):
        from skorch.exceptions import DeviceWarning
        net = net_cls(module_cls, device='cuda').initialize()

        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        net.save_params(f_params=str(p))

        with patch('torch.cuda.is_available', lambda *_: False):
            with pytest.warns(DeviceWarning) as w:
                net.load_params(f_params=str(p))

        assert w.list[0].message.args[0] == (
            'Requested to load data to CUDA but no CUDA devices '
            'are available. Loading on device "cpu" instead.')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_save_params_cuda_load_params_cpu_when_cuda_available(
            self, net_cls, module_cls, data, tmpdir):
        # Test that if we have a cuda device, we can save cuda
        # parameters and then load them to cpu
        X, y = data
        net = net_cls(module_cls, device='cuda', max_epochs=1).fit(X, y)
        p = tmpdir.mkdir('skorch').join('testmodel.pkl')
        net.save_params(f_params=str(p))

        net2 = net_cls(module_cls, device='cpu').initialize()
        net2.load_params(f_params=str(p))
        net2.predict(X)  # does not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    @pytest.mark.parametrize('parameter,name', [
        ('f_params', 'net_cuda.pt'),
        ('f_optimizer', 'optimizer_cuda.pt'),
    ])
    def test_load_cuda_params_to_cuda(
            self, parameter, name, net_cls, module_cls, data):
        net = net_cls(module_cls, device='cuda').initialize()
        # object was trained with CUDA
        kwargs = {parameter: os.path.join('skorch', 'tests', name)}
        net.load_params(**kwargs)
        net.predict(data[0])  # does not raise

    @pytest.mark.parametrize('parameter,name', [
        ('f_params', 'net_cuda.pt'),
        ('f_optimizer', 'optimizer_cuda.pt'),
    ])
    def test_load_cuda_params_to_cpu(
            self, parameter, name, net_cls, module_cls, data):
        # Note: This test will pass trivially when CUDA is available
        # but triggered a bug when CUDA is not available.
        net = net_cls(module_cls).initialize()
        # object was trained with CUDA
        kwargs = {parameter: os.path.join('skorch', 'tests', name)}
        net.load_params(**kwargs)
        net.predict(data[0])  # does not raise

    def test_save_params_with_history_file_obj(
            self, net_cls, module_cls, net_fit, tmpdir):
        net = net_cls(module_cls).initialize()

        history_before = net_fit.history

        p = tmpdir.mkdir('skorch').join('history.json')
        with open(str(p), 'w') as f:
            net_fit.save_params(f_history=f)
        del net_fit
        with open(str(p), 'r') as f:
            net.load_params(f_history=f)

        assert net.history == history_before

    @pytest.mark.parametrize('converter', [str, Path])
    def test_save_params_with_history_file_path(
            self, net_cls, module_cls, net_fit, tmpdir, converter):
        # Test loading/saving with different kinds of path representations.

        if converter is Path and sys.version < '3.6':
            # `PosixPath` cannot be `open`ed in Python < 3.6
            pytest.skip()

        net = net_cls(module_cls).initialize()

        history_before = net_fit.history

        p = tmpdir.mkdir('skorch').join('history.json')
        net_fit.save_params(f_history=converter(p))
        del net_fit
        net.load_params(f_history=converter(p))

        assert net.history == history_before

    @pytest.mark.parametrize('method, call_count', [
        ('on_train_begin', 1),
        ('on_train_end', 1),
        ('on_epoch_begin', 10),
        ('on_epoch_end', 10),
        # by default: 80/20 train/valid split
        ('on_batch_begin', (800 // 128 + 1) * 10 + (200 // 128 + 1) * 10),
        ('on_batch_end', (800 // 128 + 1) * 10 + (200 // 128 + 1) * 10),
    ])
    def test_callback_is_called(self, net_fit, method, call_count):
        # callback -2 is the mocked callback
        method = getattr(net_fit.callbacks_[-2][1], method)
        assert method.call_count == call_count
        assert method.call_args_list[0][0][0] is net_fit

    def test_history_correct_shape(self, net_fit):
        assert len(net_fit.history) == net_fit.max_epochs

    def test_history_default_keys(self, net_fit):
        expected_keys = {
            'train_loss', 'valid_loss', 'epoch', 'dur', 'batches', 'valid_acc'}
        for row in net_fit.history:
            assert expected_keys.issubset(row)

    def test_history_is_filled(self, net_fit):
        assert len(net_fit.history) == net_fit.max_epochs

    def test_set_params_works(self, net, data):
        X, y = data
        net.fit(X, y)

        assert net.module_.sequential[0].out_features == 10
        assert isinstance(net.module_.sequential[1], nn.ReLU)
        assert net.module_.sequential[3].in_features == 10
        assert np.isclose(net.lr, 0.1)

        net.set_params(
            module__hidden_units=20,
            module__nonlin=nn.Tanh(),
            lr=0.2,
        )
        net.fit(X, y)

        assert net.module_.sequential[0].out_features == 20
        assert isinstance(net.module_.sequential[1], nn.Tanh)
        assert net.module_.sequential[3].in_features == 20
        assert np.isclose(net.lr, 0.2)

    def test_set_params_then_initialize_remembers_param(
            self, net_cls, module_cls):
        net = net_cls(module_cls)

        # net does not 'forget' that params were set
        assert net.verbose != 123
        net.set_params(verbose=123)
        assert net.verbose == 123
        net.initialize()
        assert net.verbose == 123

    def test_set_params_on_callback_then_initialize_remembers_param(
            self, net_cls, module_cls):
        net = net_cls(module_cls).initialize()

        # net does not 'forget' that params were set
        assert dict(net.callbacks_)['print_log'].sink is print
        net.set_params(callbacks__print_log__sink=123)
        assert dict(net.callbacks_)['print_log'].sink == 123
        net.initialize()
        assert dict(net.callbacks_)['print_log'].sink == 123

    def test_changing_model_reinitializes_optimizer(self, net, data):
        # The idea is that we change the model using `set_params` to
        # add parameters. Since the optimizer depends on the model
        # parameters it needs to be reinitialized.
        X, y = data

        net.set_params(module__nonlin=nn.ReLU())
        net.fit(X, y)

        net.set_params(module__nonlin=nn.PReLU())
        assert isinstance(net.module_.nonlin, nn.PReLU)
        d1 = net.module_.nonlin.weight.data.clone().cpu().numpy()

        # make sure that we do not initialize again by making sure that
        # the network is initialized and by using partial_fit.
        assert net.initialized_
        net.partial_fit(X, y)
        d2 = net.module_.nonlin.weight.data.clone().cpu().numpy()

        # all newly introduced parameters should have been trained (changed)
        # by the optimizer after 10 epochs.
        assert (abs(d2 - d1) > 1e-05).all()

    def test_setting_optimizer_needs_model(self, net_cls, module_cls):
        net = net_cls(module_cls)
        assert not hasattr(net, 'module_')
        # should not break
        net.set_params(optimizer=torch.optim.SGD)

    def test_setting_lr_after_init_reflected_in_optimizer(
            self, net_cls, module_cls):
        # Fixes a bug that occurred when using set_params(lr=new_lr)
        # after initialization: The new lr was not reflected in the
        # optimizer.
        net = net_cls(module_cls).initialize()
        net.set_params(lr=10)
        assert net.lr == 10

        pg_lrs = [pg['lr'] for pg in net.optimizer_.param_groups]
        for pg_lr in pg_lrs:
            assert pg_lr == 10

    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        (
            # virtual params should prevent re-initialization
            {'optimizer__lr': 0.12, 'optimizer__momentum': 0.34},
            ("")
        ),
        (
            {'module__input_units': 12, 'module__hidden_units': 34},
            ("Re-initializing module because the following "
             "parameters were re-set: module__hidden_units, module__input_units.\n"
             "Re-initializing criterion.\n"
             "Re-initializing optimizer.")
        ),
        (
            {'criterion__reduce': False, 'criterion__size_average': True},
            ("Re-initializing criterion because the following "
             "parameters were re-set: criterion__reduce, criterion__size_average.\n"
             "Re-initializing optimizer.")
        ),
        (
            {'module__input_units': 12, 'criterion__reduce': True,
             'optimizer__momentum': 0.56},
            ("Re-initializing module because the following "
             "parameters were re-set: module__input_units.\n"
             "Re-initializing criterion.\n"
             "Re-initializing optimizer.")
        ),
    ])
    def test_reinitializing_module_optimizer_message(
            self, net_cls, module_cls, kwargs, expected, capsys):
        # When net is initialized, if module, criterion, or optimizer need to be
        # re-initialized, alert the user to the fact what parameters were
        # responsible for re-initialization. Note that when the module/criterion
        # parameters but not optimizer parameters were changed, the optimizer is
        # re-initialized but not because the optimizer parameters changed.
        net = net_cls(module_cls).initialize()
        net.set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        assert msg == expected

    @pytest.mark.parametrize('kwargs', [
        {},
        {'module__input_units': 12, 'module__hidden_units': 34},
        {'lr': 0.12},
        {'optimizer__lr': 0.12},
        {'module__input_units': 12, 'lr': 0.56},
    ])
    def test_reinitializing_module_optimizer_not_initialized_no_message(
            self, net_cls, module_cls, kwargs, capsys):
        # When net is *not* initialized, set_params on module or
        # optimizer should not trigger a message.
        net = net_cls(module_cls)
        net.set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        assert msg == ""

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, ""),  # no param, no message
        ({'lr': 0.12}, ""),  # virtual param
        ({'optimizer__lr': 0.12}, ""),  # virtual param
        ({'module__input_units': 12}, "Re-initializing optimizer."),
        ({'module__input_units': 12, 'lr': 0.56}, "Re-initializing optimizer."),
    ])
    def test_reinitializing_module_optimizer_when_initialized_message(
            self, net_cls, module_cls, kwargs, expected, capsys):
        # When the not *is* initialized, set_params on module should trigger a
        # message
        net = net_cls(module_cls).initialize()
        net.set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        # don't check the whole message since it may contain other bits not
        # tested here
        assert expected in msg

    def test_set_params_on_uninitialized_net_doesnt_initialize(self, net_cls, module_cls):
        # It used to be the case that setting a parameter on, say, the module
        # would always (re-)initialize the module, even if the whole net was not
        # initialized yet. This is unnecessary at best and can break things at
        # worst.
        net = net_cls(module_cls)
        net.set_params(module__input_units=12)
        assert not net.initialized_
        assert not hasattr(net, 'module_')

    def test_optimizer_param_groups(self, net_cls, module_cls):
        net = net_cls(
            module_cls,
            optimizer__param_groups=[
                ('sequential.0.*', {'lr': 0.1}),
                ('sequential.3.*', {'lr': 0.5}),
            ],
        )
        net.initialize()

        # two custom (1st linear, 2nd linear), one default with the
        # rest of the parameters (output).
        assert len(net.optimizer_.param_groups) == 3
        assert net.optimizer_.param_groups[0]['lr'] == 0.1
        assert net.optimizer_.param_groups[1]['lr'] == 0.5
        assert net.optimizer_.param_groups[2]['lr'] == net.lr

    def test_module_params_in_init(self, net_cls, module_cls, data):
        X, y = data

        net = net_cls(
            module=module_cls,
            module__hidden_units=20,
            module__nonlin=nn.Tanh(),
        )
        net.fit(X, y)

        assert net.module_.sequential[0].out_features == 20
        assert net.module_.sequential[3].in_features == 20
        assert isinstance(net.module_.sequential[1], nn.Tanh)

    def test_module_initialized_with_partial_module(self, net_cls, module_cls):
        net = net_cls(partial(module_cls, hidden_units=123))
        net.initialize()
        assert net.module_.sequential[0].out_features == 123

    def test_criterion_init_with_params(self, net_cls, module_cls):
        call_count = 0
        class MyCriterion(nn.Module):
            def __init__(self, spam=None):
                nonlocal call_count
                super().__init__()
                self.spam = spam
                call_count += 1

        net = net_cls(module_cls, criterion=MyCriterion, criterion__spam='eggs')
        net.initialize()
        assert call_count == 1
        assert net.criterion_.spam == 'eggs'

    def test_criterion_set_params(self, net_cls, module_cls):
        call_count = 0
        class MyCriterion(nn.Module):
            def __init__(self, spam=None):
                nonlocal call_count
                super().__init__()
                self.spam = spam
                call_count += 1

        net = net_cls(module_cls, criterion=MyCriterion)
        net.initialize()
        net.set_params(criterion__spam='eggs')
        assert call_count == 2
        assert net.criterion_.spam == 'eggs'

    def test_criterion_non_module(self, net_cls, module_cls, data):
        # test non-nn.Module classes passed as criterion
        class SimpleCriterion:
            def __call__(self, y_pred, y_true):
                return y_pred.mean()

        net = net_cls(module_cls, criterion=SimpleCriterion)
        net.initialize()
        net.fit(*data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_criterion_params_on_device(self, net_cls, module_cls, device):
        # attributes like criterion.weight should be automatically moved
        # to the Net's device.
        criterion = torch.nn.NLLLoss
        weight = torch.ones(2)
        net = net_cls(
            module_cls,
            criterion=criterion,
            criterion__weight=weight,
            device=device,
        )

        assert weight.device.type == 'cpu'
        net.initialize()
        assert net.criterion_.weight.device.type == device

    def test_callback_with_name_init_with_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(
            module_cls,
            criterion=Mock(),
            callbacks=[('cb0', mock)],
            callbacks__cb0__spam='eggs',
        )
        net.initialize()
        assert mock.initialize.call_count == 1
        assert mock.set_params.call_args_list[0][1]['spam'] == 'eggs'

    def test_callback_set_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(
            module_cls,
            criterion=Mock(),
            callbacks=[('cb0', mock)],
        )
        net.initialize()
        net.set_params(callbacks__cb0__spam='eggs')
        assert mock.initialize.call_count == 2  # callbacks are re-initialized
        assert mock.set_params.call_args_list[-1][1]['spam'] == 'eggs'

    def test_callback_name_collides_with_default(self, net_cls, module_cls):
        net = net_cls(module_cls, callbacks=[('train_loss', Mock())])
        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = ("Found duplicate user-set callback name 'train_loss'. "
                    "Use unique names to correct this.")
        assert str(exc.value) == expected

    def test_callback_same_inferred_name_twice(self, net_cls, module_cls):
        cb0 = Mock()
        cb1 = Mock()
        cb0.__class__.__name__ = 'some-name'
        cb1.__class__.__name__ = 'some-name'
        net = net_cls(module_cls, callbacks=[cb0, cb1])

        net.initialize()

        cbs = dict(net.callbacks_)
        assert 'some-name_1' in cbs
        assert 'some-name_2' in cbs
        assert cbs['some-name_1'] is cb0
        assert cbs['some-name_2'] is cb1

    def test_callback_keeps_order(self, net_cls, module_cls):
        cb0 = Mock()
        cb1 = Mock()
        cb0.__class__.__name__ = 'B-some-name'
        cb1.__class__.__name__ = 'A-some-name'
        net = net_cls(module_cls, callbacks=[cb0, cb1])

        net.initialize()

        cbs_names = [name for name, _ in net.callbacks_]
        expected_names = ['epoch_timer', 'train_loss', 'valid_loss',
                          'valid_acc', 'B-some-name', 'A-some-name',
                          'print_log']
        assert expected_names == cbs_names

    def test_callback_custom_name_is_untouched(self, net_cls, module_cls):
        callbacks = [('cb0', Mock()),
                     ('cb0', Mock())]
        net = net_cls(module_cls, callbacks=callbacks)

        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = ("Found duplicate user-set callback name 'cb0'. "
                    "Use unique names to correct this.")
        assert str(exc.value) == expected

    def test_callback_unique_naming_avoids_conflicts(
            self, net_cls, module_cls):
        # pylint: disable=invalid-name
        from skorch.callbacks import Callback

        class cb0(Callback):
            pass

        class cb0_1(Callback):
            pass

        callbacks = [cb0(), cb0(), cb0_1()]
        net = net_cls(module_cls, callbacks=callbacks)
        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = ("Assigning new callback name failed "
                    "since new name 'cb0_1' exists already.")

        assert str(exc.value) == expected

    def test_in_sklearn_pipeline(self, pipe, data):
        X, y = data
        pipe.fit(X, y)
        pipe.predict(X)
        pipe.predict_proba(X)
        pipe.set_params(net__module__hidden_units=20)

    def test_grid_search_works(self, net_cls, module_cls, data):
        net = net_cls(module_cls)
        X, y = data
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'module__hidden_units': [10, 20],
        }
        gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')
        gs.fit(X[:100], y[:100])  # for speed
        print(gs.best_score_, gs.best_params_)

    def test_change_get_loss(self, net_cls, module_cls, data):
        from skorch.utils import to_tensor

        class MyNet(net_cls):
            # pylint: disable=unused-argument
            def get_loss(self, y_pred, y_true, X=None, training=False):
                y_true = to_tensor(y_true, device='cpu')
                loss_a = torch.abs(y_true.float() - y_pred[:, 1]).mean()
                loss_b = ((y_true.float() - y_pred[:, 1]) ** 2).mean()
                if training:
                    self.history.record_batch('loss_a', to_numpy(loss_a))
                    self.history.record_batch('loss_b', to_numpy(loss_b))
                return loss_a + loss_b

        X, y = data
        net = MyNet(module_cls, max_epochs=1)
        net.fit(X, y)

        diffs = []
        all_losses = net.history[
            -1, 'batches', :, ('train_loss', 'loss_a', 'loss_b')]
        diffs = [total - a - b for total, a, b in all_losses]
        assert np.allclose(diffs, 0, atol=1e-7)

    def test_net_no_valid(self, net_cls, module_cls, data):
        net = net_cls(
            module_cls,
            max_epochs=10,
            lr=0.1,
            train_split=None,
        )
        X, y = data
        net.fit(X, y)
        assert net.history[:, 'train_loss']
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            net.history[:, 'valid_loss']

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_use_cuda_on_model(self, net_cls, module_cls):
        net_cuda = net_cls(module_cls, device='cuda')
        net_cuda.initialize()
        net_cpu = net_cls(module_cls, device='cpu')
        net_cpu.initialize()

        cpu_tensor = net_cpu.module_.sequential[0].weight.data
        assert isinstance(cpu_tensor, torch.FloatTensor)

        gpu_tensor = net_cuda.module_.sequential[0].weight.data
        assert isinstance(gpu_tensor, torch.cuda.FloatTensor)

    def test_get_params_works(self, net_cls, module_cls):
        from skorch.callbacks import EpochScoring

        net = net_cls(
            module_cls, callbacks=[('myscore', EpochScoring('myscore'))])

        params = net.get_params(deep=True)
        # test a couple of expected parameters
        assert 'verbose' in params
        assert 'module' in params
        assert 'callbacks' in params
        assert 'callbacks__print_log__sink' in params
        # not yet initialized
        assert 'callbacks__myscore__scoring' not in params

        net.initialize()
        params = net.get_params(deep=True)
        # now initialized
        assert 'callbacks__myscore__scoring' in params

    def test_get_params_with_uninit_callbacks(self, net_cls, module_cls):
        from skorch.callbacks import EpochTimer

        net = net_cls(
            module_cls,
            callbacks=[EpochTimer, ('other_timer', EpochTimer)],
        )
        # none of this raises an exception
        net = clone(net)
        net.get_params()
        net.initialize()
        net.get_params()

    def test_get_params_no_learned_params(self, net_fit):
        params = net_fit.get_params()
        params_learned = set(filter(lambda x: x.endswith('_'), params))
        assert not params_learned

    def test_clone_results_in_uninitialized_net(
            self, net_fit, data):
        X, y = data
        accuracy = accuracy_score(net_fit.predict(X), y)
        assert accuracy > ACCURACY_EXPECTED  # make sure net has learned

        net_cloned = clone(net_fit).set_params(max_epochs=0)
        net_cloned.callbacks_ = []
        net_cloned.partial_fit(X, y)
        accuracy_cloned = accuracy_score(net_cloned.predict(X), y)
        assert accuracy_cloned < ACCURACY_EXPECTED

        assert not net_cloned.history

    def test_clone_copies_parameters(self, net_cls, module_cls):
        kwargs = dict(
            module__hidden_units=20,
            lr=0.2,
            iterator_train__batch_size=123,
        )
        net = net_cls(module_cls, **kwargs)
        net_cloned = clone(net)
        params = net_cloned.get_params()
        for key, val in kwargs.items():
            assert params[key] == val

    def test_with_initialized_module(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(module_cls(), max_epochs=1)
        net.fit(X, y)

    def test_with_initialized_module_other_params(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(module_cls(), max_epochs=1, module__hidden_units=123)
        net.fit(X, y)
        weight = net.module_.sequential[0].weight.data
        assert weight.shape[0] == 123

    def test_with_initialized_module_non_default(
            self, net_cls, module_cls, data, capsys):
        X, y = data
        net = net_cls(module_cls(hidden_units=123), max_epochs=1)
        net.fit(X, y)
        weight = net.module_.sequential[0].weight.data
        assert weight.shape[0] == 123

        stdout = capsys.readouterr()[0]
        assert "Re-initializing module!" not in stdout

    def test_no_callbacks(self, net_cls, module_cls):
        net = net_cls(module_cls, callbacks="disable")
        net.initialize()
        assert net.callbacks_ == []

    def test_message_fit_with_initialized_net(
            self, net_cls, module_cls, data, capsys):
        net = net_cls(module_cls).initialize()
        net.fit(*data)
        stdout = capsys.readouterr()[0]

        msg_module = "Re-initializing module"
        assert msg_module in stdout

        msg_optimizer = "Re-initializing optimizer"
        assert msg_optimizer in stdout

        # bug: https://github.com/skorch-dev/skorch/issues/436
        not_expected = 'because the following parameters were re-set'
        assert not_expected not in stdout

    def test_with_initialized_module_partial_fit(
            self, net_cls, module_cls, data, capsys):
        X, y = data
        module = module_cls(hidden_units=123)
        net = net_cls(module, max_epochs=0)
        net.partial_fit(X, y)

        for p0, p1 in zip(module.parameters(), net.module_.parameters()):
            assert p0.data.shape == p1.data.shape
            assert (p0 == p1).data.all()

        stdout = capsys.readouterr()[0]
        assert "Re-initializing module!" not in stdout

    def test_with_initialized_module_warm_start(
            self, net_cls, module_cls, data, capsys):
        X, y = data
        module = module_cls(hidden_units=123)
        net = net_cls(module, max_epochs=0, warm_start=True)
        net.partial_fit(X, y)

        for p0, p1 in zip(module.parameters(), net.module_.parameters()):
            assert p0.data.shape == p1.data.shape
            assert (p0 == p1).data.all()

        stdout = capsys.readouterr()[0]
        assert "Re-initializing module!" not in stdout

    def test_with_initialized_sequential(self, net_cls, data, capsys):
        X, y = data
        module = nn.Sequential(
            nn.Linear(X.shape[1], 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Softmax(dim=-1),
        )
        net = net_cls(module, max_epochs=1)
        net.fit(X, y)

        stdout = capsys.readouterr()[0]
        assert "Re-initializing module!" not in stdout

    def test_call_fit_twice_retrains(self, net_cls, module_cls, data):
        # test that after second fit call, even without entering the
        # fit loop, parameters have changed (because the module was
        # re-initialized)
        X, y = data[0][:100], data[1][:100]
        net = net_cls(module_cls, warm_start=False).fit(X, y)
        params_before = net.module_.parameters()

        net.max_epochs = 0
        net.fit(X, y)
        params_after = net.module_.parameters()

        assert not net.history
        for p0, p1 in zip(params_before, params_after):
            assert (p0 != p1).data.any()

    def test_call_fit_twice_warmstart(self, net_cls, module_cls, data):
        X, y = data[0][:100], data[1][:100]
        net = net_cls(module_cls, warm_start=True).fit(X, y)
        params_before = net.module_.parameters()

        net.max_epochs = 0
        net.fit(X, y)
        params_after = net.module_.parameters()

        assert len(net.history) == 10
        for p0, p1 in zip(params_before, params_after):
            assert (p0 == p1).data.all()

    def test_partial_fit_first_call(self, net_cls, module_cls, data):
        # It should be possible to partial_fit without calling fit first.
        X, y = data[0][:100], data[1][:100]
        # does not raise
        net_cls(module_cls, warm_start=True).partial_fit(X, y)

    def test_call_partial_fit_after_fit(self, net_cls, module_cls, data):
        X, y = data[0][:100], data[1][:100]
        net = net_cls(module_cls, warm_start=False).fit(X, y)
        params_before = net.module_.parameters()

        net.max_epochs = 0
        net.partial_fit(X, y)
        params_after = net.module_.parameters()

        assert len(net.history) == 10
        for p0, p1 in zip(params_before, params_after):
            assert (p0 == p1).data.all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_binary_classification_with_cuda(self, net_cls, module_cls, data):
        X, y = data
        assert y.ndim == 1
        assert set(y) == {0, 1}

        net = net_cls(module_cls, max_epochs=1, device='cuda')
        # does not raise
        net.fit(X, y)

    def test_net_initialized_with_custom_dataset_args(
            self, net_cls, module_cls, data, dataset_cls):
        side_effect = []

        class MyDataset(dataset_cls):
            def __init__(self, *args, foo, **kwargs):
                super().__init__(*args, **kwargs)
                side_effect.append(foo)

        net = net_cls(
            module_cls,
            dataset=MyDataset,
            dataset__foo=123,
            max_epochs=1,
        )
        net.fit(*data)
        assert side_effect == [123]

    @pytest.mark.xfail(raises=ValueError)
    def test_net_initialized_with_initalized_dataset(
            self, net_cls, module_cls, data, dataset_cls):
        net = net_cls(
            module_cls,
            dataset=dataset_cls(*data),
            max_epochs=1,

            # Disable caching to highlight the issue with this
            # test case (mismatching size between y values)
            callbacks__valid_acc__use_caching=False,
        )
        # FIXME: When dataset is initialized, X and y do not matter
        # anymore
        net.fit(*data)  # should not raise

    def test_net_initialized_with_partialed_dataset(
            self, net_cls, module_cls, data, dataset_cls):
        X, y = data
        net = net_cls(
            module_cls,
            dataset=partial(dataset_cls, length=len(y)),
            train_split=None,
            max_epochs=1,
        )
        net.fit(X, y)  # does not raise

    def test_net_initialized_with_initalized_dataset_and_kwargs_raises(
            self, net_cls, module_cls, data, dataset_cls):
        net = net_cls(
            module_cls,
            dataset=dataset_cls(*data),
            dataset__foo=123,
            max_epochs=1,
        )
        with pytest.raises(TypeError) as exc:
            net.fit(*data)

        expected = ("Trying to pass an initialized Dataset while passing "
                    "Dataset arguments ({'foo': 123}) is not allowed.")
        assert exc.value.args[0] == expected

    def test_repr_uninitialized_works(self, net_cls, module_cls):
        net = net_cls(
            module_cls,
            module__hidden_units=55,
        )
        result = net.__repr__()
        expected = """<class 'skorch.classifier.NeuralNetClassifier'>[uninitialized](
  module={},
  module__hidden_units=55,
)""".format(module_cls)
        assert result == expected

    def test_repr_initialized_works(self, net_cls, module_cls):
        net = net_cls(
            module_cls,
            module__hidden_units=42,
        )
        net.initialize()
        result = net.__repr__()
        expected = """<class 'skorch.classifier.NeuralNetClassifier'>[initialized](
  module_=MLPModule(
    (nonlin): ReLU()
    (output_nonlin): Softmax()
    (sequential): Sequential(
      (0): Linear(in_features=20, out_features=42, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.5)
      (3): Linear(in_features=42, out_features=42, bias=True)
      (4): ReLU()
      (5): Dropout(p=0.5)
      (6): Linear(in_features=42, out_features=2, bias=True)
      (7): Softmax()
    )
  ),
)"""
        if LooseVersion(torch.__version__) >= '1.2':
            expected = expected.replace("Softmax()", "Softmax(dim=-1)")
            expected = expected.replace("Dropout(p=0.5)",
                                        "Dropout(p=0.5, inplace=False)")
        assert result == expected

    def test_repr_fitted_works(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(
            module_cls,
            module__hidden_units=11,
            module__nonlin=nn.PReLU(),
        )
        net.fit(X[:50], y[:50])
        result = net.__repr__()
        expected = """<class 'skorch.classifier.NeuralNetClassifier'>[initialized](
  module_=MLPModule(
    (nonlin): PReLU(num_parameters=1)
    (output_nonlin): Softmax()
    (sequential): Sequential(
      (0): Linear(in_features=20, out_features=11, bias=True)
      (1): PReLU(num_parameters=1)
      (2): Dropout(p=0.5)
      (3): Linear(in_features=11, out_features=11, bias=True)
      (4): PReLU(num_parameters=1)
      (5): Dropout(p=0.5)
      (6): Linear(in_features=11, out_features=2, bias=True)
      (7): Softmax()
    )
  ),
)"""
        if LooseVersion(torch.__version__) >= '1.2':
            expected = expected.replace("Softmax()", "Softmax(dim=-1)")
            expected = expected.replace("Dropout(p=0.5)",
                                        "Dropout(p=0.5, inplace=False)")
        assert result == expected

    def test_fit_params_passed_to_module(self, net_cls, data):
        from skorch.toy import MLPModule

        X, y = data
        side_effect = []

        class FPModule(MLPModule):
            # pylint: disable=arguments-differ
            def forward(self, X, **fit_params):
                side_effect.append(fit_params)
                return super().forward(X)

        net = net_cls(FPModule, max_epochs=1, batch_size=50, train_split=None)
        # remove callbacks to have better control over side_effect
        net.initialize()
        net.callbacks_ = []
        net.fit(X[:100], y[:100], foo=1, bar=2)
        net.fit(X[:100], y[:100], bar=3, baz=4)

        assert len(side_effect) == 4  # 2 epochs à 2 batches
        assert side_effect[0] == dict(foo=1, bar=2)
        assert side_effect[1] == dict(foo=1, bar=2)
        assert side_effect[2] == dict(bar=3, baz=4)
        assert side_effect[3] == dict(bar=3, baz=4)

    def test_fit_params_passed_to_module_in_pipeline(self, net_cls, data):
        from skorch.toy import MLPModule

        X, y = data
        side_effect = []

        class FPModule(MLPModule):
            # pylint: disable=arguments-differ
            def forward(self, X, **fit_params):
                side_effect.append(fit_params)
                return super().forward(X)

        net = net_cls(FPModule, max_epochs=1, batch_size=50, train_split=None)
        net.initialize()
        net.callbacks_ = []
        pipe = Pipeline([
            ('net', net),
        ])
        pipe.fit(X[:100], y[:100], net__foo=1, net__bar=2)
        pipe.fit(X[:100], y[:100], net__bar=3, net__baz=4)

        assert len(side_effect) == 4  # 2 epochs à 2 batches
        assert side_effect[0] == dict(foo=1, bar=2)
        assert side_effect[1] == dict(foo=1, bar=2)
        assert side_effect[2] == dict(bar=3, baz=4)
        assert side_effect[3] == dict(bar=3, baz=4)

    def test_fit_params_passed_to_train_split(self, net_cls, data):
        from skorch.toy import MLPModule

        X, y = data
        side_effect = []

        # pylint: disable=unused-argument
        def fp_train_split(dataset, y=None, **fit_params):
            side_effect.append(fit_params)
            return dataset, dataset

        class FPModule(MLPModule):
            # pylint: disable=unused-argument,arguments-differ
            def forward(self, X, **fit_params):
                return super().forward(X)

        net = net_cls(
            FPModule,
            max_epochs=1,
            batch_size=50,
            train_split=fp_train_split,
        )
        net.initialize()
        net.callbacks_ = []
        net.fit(X[:100], y[:100], foo=1, bar=2)
        net.fit(X[:100], y[:100], bar=3, baz=4)

        assert len(side_effect) == 2  # 2 epochs
        assert side_effect[0] == dict(foo=1, bar=2)
        assert side_effect[1] == dict(bar=3, baz=4)

    def test_data_dict_and_fit_params(self, net_cls, data):
        from skorch.toy import MLPModule

        X, y = data

        class FPModule(MLPModule):
            # pylint: disable=unused-argument,arguments-differ
            def forward(self, X0, X1, **fit_params):
                assert fit_params.get('foo') == 3
                return super().forward(X0)

        net = net_cls(FPModule, max_epochs=1, batch_size=50, train_split=None)
        # does not raise
        net.fit({'X0': X, 'X1': X}, y, foo=3)

    def test_data_dict_and_fit_params_conflicting_names_raises(
            self, net_cls, data):
        from skorch.toy import MLPModule

        X, y = data

        class FPModule(MLPModule):
            # pylint: disable=unused-argument,arguments-differ
            def forward(self, X0, X1, **fit_params):
                return super().forward(X0)

        net = net_cls(FPModule, max_epochs=1, batch_size=50, train_split=None)

        with pytest.raises(ValueError) as exc:
            net.fit({'X0': X, 'X1': X}, y, X1=3)

        expected = "X and fit_params contain duplicate keys: X1"
        assert exc.value.args[0] == expected

    def test_fit_with_dataset(self, net_cls, module_cls, data, dataset_cls):
        ds = dataset_cls(*data)
        net = net_cls(module_cls, max_epochs=1)
        net.fit(ds, data[1])
        for key in ('train_loss', 'valid_loss', 'valid_acc'):
            assert key in net.history[-1]

    def test_predict_with_dataset(self, net_cls, module_cls, data, dataset_cls):
        ds = dataset_cls(*data)
        net = net_cls(module_cls).initialize()
        y_pred = net.predict(ds)
        y_proba = net.predict_proba(ds)

        assert y_pred.shape[0] == len(ds)
        assert y_proba.shape[0] == len(ds)

    def test_fit_with_dataset_X_y_inaccessible_does_not_raise(
            self, net_cls, module_cls, data):
        class MyDataset(torch.utils.data.Dataset):
            """Dataset with inaccessible X and y"""
            def __init__(self, X, y):
                self.xx = X  # incorrect attribute name
                self.yy = y  # incorrect attribute name

            def __len__(self):
                return len(self.xx)

            def __getitem__(self, i):
                return self.xx[i], self.yy[i]

        ds = MyDataset(*data)
        net = net_cls(module_cls, max_epochs=1)
        net.fit(ds, data[1])  # does not raise

    def test_fit_with_dataset_without_explicit_y(
            self, net_cls, module_cls, dataset_cls, data):
        from skorch.dataset import CVSplit

        net = net_cls(
            module_cls,
            max_epochs=1,
            train_split=CVSplit(stratified=False),
        )
        ds = dataset_cls(*data)
        net.fit(ds, None)  # does not raise
        for key in ('train_loss', 'valid_loss', 'valid_acc'):
            assert key in net.history[-1]

    def test_fit_with_dataset_stratified_without_explicit_y_raises(
            self, net_cls, module_cls, dataset_cls, data):
        from skorch.dataset import CVSplit

        net = net_cls(
            module_cls,
            train_split=CVSplit(stratified=True),
        )
        ds = dataset_cls(*data)
        with pytest.raises(ValueError) as exc:
            net.fit(ds, None)

        msg = "Stratified CV requires explicitly passing a suitable y."
        assert exc.value.args[0] == msg

    @pytest.fixture
    def dataset_1_item(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, i):
                return 0.0
        return Dataset

    def test_fit_with_dataset_one_item_error(
            self, net_cls, module_cls, dataset_1_item):
        net = net_cls(module_cls, train_split=None)
        with pytest.raises(ValueError) as exc:
            net.fit(dataset_1_item(), None)

        msg = ("You are using a non-skorch dataset that returns 1 value. "
               "Remember that for skorch, Dataset.__getitem__ must return "
               "exactly 2 values, X and y (more info: "
               "https://skorch.readthedocs.io/en/stable/user/dataset.html).")
        assert exc.value.args[0] == msg

    def test_predict_with_dataset_one_item_error(
            self, net_cls, module_cls, dataset_1_item):
        net = net_cls(module_cls, train_split=None).initialize()
        with pytest.raises(ValueError) as exc:
            net.predict(dataset_1_item())

        msg = ("You are using a non-skorch dataset that returns 1 value. "
               "Remember that for skorch, Dataset.__getitem__ must return "
               "exactly 2 values, X and y (more info: "
               "https://skorch.readthedocs.io/en/stable/user/dataset.html).")
        assert exc.value.args[0] == msg

    @pytest.fixture
    def dataset_3_items(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, i):
                return 0.0, 0.0, 0.0
        return Dataset

    def test_fit_with_dataset_three_items_error(
            self, net_cls, module_cls, dataset_3_items):
        net = net_cls(module_cls, train_split=None)
        with pytest.raises(ValueError) as exc:
            net.fit(dataset_3_items(), None)

        msg = ("You are using a non-skorch dataset that returns 3 values. "
               "Remember that for skorch, Dataset.__getitem__ must return "
               "exactly 2 values, X and y (more info: "
               "https://skorch.readthedocs.io/en/stable/user/dataset.html).")
        assert exc.value.args[0] == msg

    def test_predict_with_dataset_three_items_error(
            self, net_cls, module_cls, dataset_3_items):
        net = net_cls(module_cls, train_split=None).initialize()
        with pytest.raises(ValueError) as exc:
            net.predict(dataset_3_items())

        msg = ("You are using a non-skorch dataset that returns 3 values. "
               "Remember that for skorch, Dataset.__getitem__ must return "
               "exactly 2 values, X and y (more info: "
               "https://skorch.readthedocs.io/en/stable/user/dataset.html).")
        assert exc.value.args[0] == msg

    @pytest.fixture
    def multiouput_net(self, net_cls, multiouput_module):
        return net_cls(multiouput_module).initialize()

    def test_multioutput_forward_iter(self, multiouput_net, data):
        X = data[0]
        y_infer = next(multiouput_net.forward_iter(X))

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        assert y_infer[0].shape[0] == min(len(X), multiouput_net.batch_size)

    def test_multioutput_forward(self, multiouput_net, data):
        X = data[0]
        n = len(X)
        y_infer = multiouput_net.forward(X)

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        for arr in y_infer:
            assert is_torch_data_type(arr)

        # Expecting full output: (number of samples, number of output units)
        assert y_infer[0].shape == (n, 2)
        # Expecting only column 0: (number of samples,)
        assert y_infer[1].shape == (n,)
        # Expecting only every other row: (number of samples/2, number
        # of output units)
        assert y_infer[2].shape == (n // 2, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_multioutput_forward_device_gpu(self, multiouput_net, data):
        X = data[0]
        y_infer = multiouput_net.forward(X, device='cuda:0')

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        for arr in y_infer:
            assert arr.is_cuda

    def test_multioutput_predict(self, multiouput_net, data):
        X = data[0]
        n = len(X)

        # does not raise
        y_pred = multiouput_net.predict(X)

        # Expecting only 1 column containing predict class:
        # (number of samples,)
        assert y_pred.shape == (n,)
        assert set(y_pred) == {0, 1}

    def test_multiouput_predict_proba(self, multiouput_net, data):
        X = data[0]
        n = len(X)

        # does not raise
        y_proba = multiouput_net.predict_proba(X)

        # Expecting full output: (number of samples, number of output units)
        assert y_proba.shape == (n, 2)
        # Probabilities, hence these limits
        assert y_proba.min() >= 0
        assert y_proba.max() <= 1

    def test_setting_callback_possible(self, net_cls, module_cls):
        from skorch.callbacks import EpochTimer, PrintLog

        net = net_cls(module_cls, callbacks=[('mycb', PrintLog())])
        net.initialize()

        assert isinstance(dict(net.callbacks_)['mycb'], PrintLog)

        net.set_params(callbacks__mycb=EpochTimer())
        assert isinstance(dict(net.callbacks_)['mycb'], EpochTimer)

    def test_setting_callback_default_possible(self, net_cls, module_cls):
        from skorch.callbacks import EpochTimer, PrintLog

        net = net_cls(module_cls)
        net.initialize()

        assert isinstance(dict(net.callbacks_)['print_log'], PrintLog)

        net.set_params(callbacks__print_log=EpochTimer())
        assert isinstance(dict(net.callbacks_)['print_log'], EpochTimer)

    def test_setting_callback_to_none_possible(self, net_cls, module_cls, data):
        from skorch.callbacks import Callback

        X, y = data[0][:30], data[1][:30]  # accelerate test
        side_effects = []

        class DummyCallback(Callback):
            def __init__(self, i):
                self.i = i

            # pylint: disable=unused-argument, arguments-differ
            def on_epoch_end(self, *args, **kwargs):
                side_effects.append(self.i)

        net = net_cls(
            module_cls,
            max_epochs=2,
            callbacks=[
                ('cb0', DummyCallback(0)),
                ('cb1', DummyCallback(1)),
                ('cb2', DummyCallback(2)),
            ],
        )
        net.fit(X, y)

        # all 3 callbacks write to output twice
        assert side_effects == [0, 1, 2, 0, 1, 2]

        # deactivate cb1
        side_effects.clear()
        net.set_params(callbacks__cb1=None)
        net.fit(X, y)

        assert side_effects == [0, 2, 0, 2]

    def test_setting_callback_to_none_and_more_params_during_init_raises(
            self, net_cls, module_cls):
        # if a callback is set to None, setting more params for it
        # should not work
        net = net_cls(
            module_cls, callbacks__print_log=None, callbacks__print_log__sink=1)

        with pytest.raises(ValueError) as exc:
            net.initialize()

        msg = ("Trying to set a parameter for callback print_log "
               "which does not exist.")
        assert exc.value.args[0] == msg

    def test_setting_callback_to_none_and_more_params_later_raises(
            self, net_cls, module_cls):
        # this should work
        net = net_cls(module_cls)
        net.set_params(callbacks__print_log__sink=123)
        net.set_params(callbacks__print_log=None)

        net = net_cls(module_cls)
        net.set_params(callbacks__print_log=None)
        with pytest.raises(ValueError) as exc:
            net.set_params(callbacks__print_log__sink=123)

        msg = ("Trying to set a parameter for callback print_log "
               "which does not exist.")
        assert exc.value.args[0] == msg

    def test_set_params_on_init_net_normal_param_works(self, net_cls, module_cls):
        # setting "normal" arguments like max_epoch works on an initialized net
        net = net_cls(module_cls).initialize()
        net.set_params(max_epochs=3, callbacks=[])  # does not raise
        net.initialize()

    def test_set_params_with_unknown_key_raises(self, net):
        with pytest.raises(ValueError) as exc:
            net.set_params(foo=123)

        # TODO: check error message more precisely, depending on what
        # the intended message should be from sklearn side
        assert exc.value.args[0].startswith('Invalid parameter foo for')

    @pytest.fixture()
    def sequence_module_cls(self):
        """Simple sequence model with variable size dim 1."""
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(1, 1)
            # pylint: disable=arguments-differ
            def forward(self, x):
                n = np.random.randint(1, 4)
                y = self.l(x.float())
                return torch.randn(1, n, 2) + 0 * y
        return Mod

    def test_net_variable_prediction_lengths(
            self, net_cls, sequence_module_cls):
        # neural net should work fine with fixed y_true but varying y_pred
        # sequences.
        X = np.array([1, 5, 3, 6, 2])
        y = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0]])
        X, y = X[:, np.newaxis], y[:, :, np.newaxis]
        X, y = X.astype('float32'), y.astype('float32')

        net = net_cls(
            sequence_module_cls,
            batch_size=1,
            max_epochs=2,
            train_split=None,
        )

        # Mock loss function
        # pylint: disable=unused-argument
        def loss_fn(y_pred, y_true, **kwargs):
            return y_pred[:, 0, 0]
        net.get_loss = loss_fn

        net.fit(X, y)

    def test_net_variable_label_lengths(self, net_cls, sequence_module_cls):
        # neural net should work fine with varying y_true sequences.
        X = np.array([1, 5, 3, 6, 2])
        y = np.array([[1], [1, 0, 1], [1, 1], [1, 1, 0], [1, 0]])
        X = X[:, np.newaxis].astype('float32')
        y = np.array([np.array(n, dtype='float32')[:, np.newaxis] for n in y])

        net = net_cls(
            sequence_module_cls,
            batch_size=1,
            max_epochs=2,
            train_split=None,
        )

        # Mock loss function
        # pylint: disable=unused-argument
        def loss_fn(y_pred, y_true, **kwargs):
            return y_pred[:, 0, 0]
        net.get_loss = loss_fn

        # check_data complains about y.shape = (n,) but
        # we know that it is actually (n, m) with m in [1;3].
        net.check_data = lambda *_, **kw: None
        net.fit(X, y)

    def test_no_grad_during_validation(self, net_cls, module_cls, data):
        """Test that gradient is only calculated during training step,
        not validation step."""

        # pylint: disable=unused-argument
        def check_grad(*args, loss, training, **kwargs):
            if training:
                assert loss.requires_grad
            else:
                assert not loss.requires_grad

        mock_cb = Mock(on_batch_end=check_grad)
        net = net_cls(module_cls, max_epochs=1, callbacks=[mock_cb])
        net.fit(*data)

    def test_callback_on_grad_computed(self, net_cls, module_cls, data):
        module = module_cls()
        expected_names = set(name for name, _ in module.named_parameters())

        def on_grad_computed(*args, named_parameters, **kwargs):
            names = set(name for name, _ in named_parameters)
            assert expected_names == names

        mock_cb = Mock(on_grad_computed=on_grad_computed)
        net = net_cls(module, max_epochs=1, callbacks=[mock_cb])
        net.fit(*data)

    @pytest.mark.parametrize('training', [True, False])
    def test_no_grad_during_evaluation_unless_training(
            self, net_cls, module_cls, data, training):
        """Test that gradient is only calculated in training mode
        during evaluation step."""
        from skorch.utils import to_tensor

        net = net_cls(module_cls).initialize()
        Xi = to_tensor(data[0][:3], device='cpu')
        batch = Xi, None
        y_eval = net.evaluation_step(batch, training=training)

        assert y_eval.requires_grad is training

    @pytest.mark.parametrize(
        'net_kwargs,expected_train_batch_size,expected_valid_batch_size',
        [
            ({'batch_size': -1}, 800, 200),
            ({'iterator_train__batch_size': -1}, 800, 128),
            ({'iterator_valid__batch_size': -1}, 128, 200),
        ]
    )
    def test_batch_size_neg_1_uses_whole_dataset(
            self, net_cls, module_cls, data, net_kwargs,
            expected_train_batch_size, expected_valid_batch_size):

        train_loader_mock = Mock(side_effect=torch.utils.data.DataLoader)
        valid_loader_mock = Mock(side_effect=torch.utils.data.DataLoader)

        net = net_cls(module_cls, max_epochs=1,
                      iterator_train=train_loader_mock,
                      iterator_valid=valid_loader_mock,
                      **net_kwargs)
        net.fit(*data)

        train_batch_size = net.history[:, 'batches', :, 'train_batch_size'][0][0]
        valid_batch_size = net.history[:, 'batches', :, 'valid_batch_size'][0][0]

        assert train_batch_size == expected_train_batch_size
        assert valid_batch_size == expected_valid_batch_size

        # pylint: disable=unsubscriptable-object
        train_kwargs = train_loader_mock.call_args[1]
        valid_kwargs = valid_loader_mock.call_args[1]
        assert train_kwargs['batch_size'] == expected_train_batch_size
        assert valid_kwargs['batch_size'] == expected_valid_batch_size

    @pytest.mark.parametrize('batch_size', [40, 100])
    def test_batch_count(self, net_cls, module_cls, data, batch_size):

        net = net_cls(module_cls, max_epochs=1, batch_size=batch_size)
        X, y = data
        net.fit(X, y)

        train_batch_count = int(0.8 * len(X)) / batch_size
        valid_batch_count = int(0.2 * len(X)) / batch_size

        assert net.history[:, "train_batch_count"] == [train_batch_count]
        assert net.history[:, "valid_batch_count"] == [valid_batch_count]

    @flaky(max_runs=5)
    def test_fit_lbfgs_optimizer(self, net_cls, module_cls, data):
        # need to randomize the seed, otherwise flaky always runs with
        # the exact same seed
        torch.manual_seed(int(time.time()))
        X, y = data
        net = net_cls(
            module_cls,
            optimizer=torch.optim.LBFGS,
            lr=1.0,
            batch_size=-1,
        )
        net.fit(X, y)

        last_epoch = net.history[-1]
        assert last_epoch['train_loss'] < 1.0
        assert last_epoch['valid_loss'] < 1.0
        assert last_epoch['valid_acc'] > 0.75

    def test_accumulator_that_returns_last_value(
            self, net_cls, module_cls, data):
        # We define an optimizer that calls the step function 3 times
        # and an accumulator that returns the last of those calls. We
        # then test that the correct values were stored.
        from skorch.utils import FirstStepAccumulator

        side_effect = []

        class SGD3Calls(torch.optim.SGD):
            def step(self, closure=None):
                for _ in range(3):
                    loss = super().step(closure)
                    side_effect.append(float(loss))

        class MyAccumulator(FirstStepAccumulator):
            """Accumulate all steps and return the last."""
            def store_step(self, step):
                if self.step is None:
                    self.step = [step]
                else:
                    self.step.append(step)

            def get_step(self):
                # Losses should only ever be retrieved after storing 3
                # times.
                assert len(self.step) == 3
                return self.step[-1]

        X, y = data
        max_epochs = 2
        batch_size = 100
        net = net_cls(
            module_cls,
            optimizer=SGD3Calls,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=None,
        )
        net.get_train_step_accumulator = MyAccumulator
        net.fit(X, y)

        # Number of loss calculations is total number of batches x 3.
        num_batches_per_epoch = int(np.ceil(len(y) / batch_size))
        expected_calls = 3 * num_batches_per_epoch * max_epochs
        assert len(side_effect) == expected_calls

        # Every 3rd loss calculation (i.e. the last per call) should
        # be stored in the history.
        expected_losses = list(
            flatten(net.history[:, 'batches', :, 'train_loss']))
        assert np.allclose(side_effect[2::3], expected_losses)

    @pytest.fixture
    def predefined_split(self):
        from skorch.helper import predefined_split
        return predefined_split

    def test_predefined_split(
            self, net_cls, module_cls, data, predefined_split, dataset_cls):
        train_loader_mock = Mock(side_effect=torch.utils.data.DataLoader)
        valid_loader_mock = Mock(side_effect=torch.utils.data.DataLoader)

        train_ds = dataset_cls(*data)
        valid_ds = dataset_cls(*data)

        net = net_cls(
            module_cls, max_epochs=1,
            iterator_train=train_loader_mock,
            iterator_valid=valid_loader_mock,
            train_split=predefined_split(valid_ds)
        )

        net.fit(train_ds, None)

        # pylint: disable=unsubscriptable-object
        train_loader_ds = train_loader_mock.call_args[0][0]
        valid_loader_ds = valid_loader_mock.call_args[0][0]

        assert train_loader_ds == train_ds
        assert valid_loader_ds == valid_ds

    def test_predefined_split_with_y(
            self, net_cls, module_cls, data, predefined_split, dataset_cls):
        # A change in the signature of utils._make_split in #646 led
        # to a bug reported in #681, namely `TypeError: _make_split()
        # got multiple values for argument 'valid_ds'`. This is a test
        # for the bug.
        X, y = data
        X_train, y_train, X_valid, y_valid = X[:800], y[:800], X[800:], y[800:]
        valid_ds = dataset_cls(X_valid, y_valid)
        net = net_cls(
            module_cls,
            max_epochs=1,
            train_split=predefined_split(valid_ds),
        )
        net.fit(X_train, y_train)

    def test_set_lr_at_runtime_doesnt_reinitialize(self, net_fit):
        with patch('skorch.NeuralNet.initialize_optimizer') as f:
            net_fit.set_params(lr=0.9)
        assert not f.called

    def test_set_lr_at_runtime_sets_lr(self, net_fit):
        new_lr = net_fit.lr + 1
        net_fit.set_params(lr=new_lr)

        assert net_fit.lr == new_lr
        assert net_fit.optimizer_.param_groups[0]['lr'] == new_lr

    def test_set_lr_at_runtime_sets_lr_via_pgroup_0(self, net_fit):
        new_lr = net_fit.lr + 1
        net_fit.set_params(optimizer__param_groups__0__lr=new_lr)

        # note that setting group does not set global lr
        assert net_fit.lr != new_lr
        assert net_fit.optimizer_.param_groups[0]['lr'] == new_lr

    def test_set_lr_at_runtime_sets_lr_pgroups(self, net_cls, module_cls, data):
        lr_pgroup_0 = 0.1
        lr_pgroup_1 = 0.2
        lr_pgroup_0_new = 0.3
        lr_pgroup_1_new = 0.4

        net = net_cls(
            module_cls,
            lr=lr_pgroup_1,
            max_epochs=1,
            optimizer__param_groups=[
                ('sequential.0.*', {'lr': lr_pgroup_0}),
            ])
        net.fit(*data)

        # optimizer__param_groups=[g1] will create
        # - param group 0 matching the definition of g1
        # - param group 1 matching all other parameters
        assert net.optimizer_.param_groups[0]['lr'] == lr_pgroup_0
        assert net.optimizer_.param_groups[1]['lr'] == lr_pgroup_1

        net.set_params(optimizer__param_groups__0__lr=lr_pgroup_0_new)
        net.set_params(optimizer__param_groups__1__lr=lr_pgroup_1_new)

        assert net.optimizer_.param_groups[0]['lr'] == lr_pgroup_0_new
        assert net.optimizer_.param_groups[1]['lr'] == lr_pgroup_1_new

    def test_criterion_training_set_correctly(self, net_cls, module_cls, data):
        # check that criterion's training attribute is set correctly

        X, y = data[0][:50], data[1][:50]  # don't need all the data
        side_effect = []

        class MyCriterion(nn.NLLLoss):
            """Criterion that records its training attribute"""
            def forward(self, *args, **kwargs):
                side_effect.append(self.training)
                return super().forward(*args, **kwargs)

        net = net_cls(module_cls, criterion=MyCriterion, max_epochs=1)
        net.fit(X, y)

        # called once with training=True for train step, once with
        # training=False for validation step
        assert side_effect == [True, False]

        net.partial_fit(X, y)
        # same logic as before
        assert side_effect == [True, False, True, False]

    def test_criterion_is_not_a_torch_module(self, net_cls, module_cls, data):
        X, y = data[0][:50], data[1][:50]  # don't need all the data

        def my_criterion():
            return torch.nn.functional.nll_loss

        net = net_cls(module_cls, criterion=my_criterion, max_epochs=1)
        net.fit(X, y)  # does not raise

    @pytest.mark.parametrize('acc_steps', [1, 2, 3, 5, 10])
    def test_gradient_accumulation(self, net_cls, module_cls, data, acc_steps):
        # Test if gradient accumulation technique is possible,
        # i.e. performing a weight update only every couple of
        # batches.
        mock_optimizer = Mock()

        class GradAccNet(net_cls):
            """Net that accumulates gradients"""
            def __init__(self, *args, acc_steps=acc_steps, **kwargs):
                super().__init__(*args, **kwargs)
                self.acc_steps = acc_steps

            def initialize_optimizer(self):
                # This is not necessary for gradient accumulation but
                # only for testing purposes
                super().initialize_optimizer()
                # pylint: disable=access-member-before-definition
                self.true_optimizer_ = self.optimizer_
                mock_optimizer.step.side_effect = self.true_optimizer_.step
                mock_optimizer.zero_grad.side_effect = self.true_optimizer_.zero_grad
                self.optimizer_ = mock_optimizer
                return self

            def get_loss(self, *args, **kwargs):
                loss = super().get_loss(*args, **kwargs)
                # because only every nth step is optimized
                return loss / self.acc_steps

            def train_step(self, batch, **fit_params):
                """Perform gradient accumulation

                Only optimize every 2nd batch.

                """
                # note that n_train_batches starts at 1 for each epoch
                n_train_batches = len(self.history[-1, 'batches'])
                step = self.train_step_single(batch, **fit_params)

                if n_train_batches % self.acc_steps == 0:
                    self.optimizer_.step()
                    self.optimizer_.zero_grad()
                return step

        max_epochs = 5
        net = GradAccNet(module_cls, max_epochs=max_epochs)
        X, y = data
        net.fit(X, y)

        n = len(X) * 0.8  # number of training samples
        b = np.ceil(n / net.batch_size)  # batches per epoch
        s = b // acc_steps  # number of acc steps per epoch
        calls_total = s * max_epochs
        calls_step = mock_optimizer.step.call_count
        calls_zero_grad = mock_optimizer.zero_grad.call_count
        assert calls_total == calls_step == calls_zero_grad

    def test_setattr_custom_module(self, net_cls, module_cls):
        # creating a custom module should result in its regiestration
        net = net_cls(module_cls).initialize()
        assert 'mymodule' not in net.prefixes_
        assert 'mymodule_' not in net.cuda_dependent_attributes_
        assert 'mymodule' not in net.modules_

        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.mymodule_ = module_cls()
                return self

        net = MyNet(module_cls).initialize()
        assert 'mymodule' in net.prefixes_
        assert 'mymodule_' in net.cuda_dependent_attributes_
        assert 'mymodule' in net.modules_

        del net.mymodule_
        assert 'mymodule' not in net.prefixes_
        assert 'mymodule_' not in net.cuda_dependent_attributes_
        assert 'mymodule' not in net.modules_

    def test_setattr_custom_criterion(self, net_cls, module_cls):
        # creating a custom criterion should result in its regiestration
        net = net_cls(module_cls).initialize()
        assert 'mycriterion' not in net.prefixes_
        assert 'mycriterion_' not in net.cuda_dependent_attributes_
        assert 'mycriterion' not in net.criteria_

        class MyNet(net_cls):
            def initialize_criterion(self):
                super().initialize_criterion()
                self.mycriterion_ = module_cls()
                return self

        net = MyNet(module_cls).initialize()
        assert 'mycriterion' in net.prefixes_
        assert 'mycriterion_' in net.cuda_dependent_attributes_
        assert 'mycriterion' in net.criteria_

        del net.mycriterion_
        assert 'mycriterion' not in net.prefixes_
        assert 'mycriterion_' not in net.cuda_dependent_attributes_
        assert 'mycriterion' not in net.criteria_

    def test_setattr_custom_optimizer(self, net_cls, module_cls):
        # creating a custom optimizer should result in its regiestration
        net = net_cls(module_cls).initialize()
        assert 'myoptimizer' not in net.prefixes_
        assert 'myoptimizer_' not in net.cuda_dependent_attributes_
        assert 'myoptimizer' not in net.prefixes_

        class MyNet(net_cls):
            def initialize_optimizer(self):
                super().initialize_optimizer()
                self.myoptimizer_ = torch.optim.SGD(self.module_.parameters(), lr=1)
                return self

        net = MyNet(module_cls).initialize()
        assert 'myoptimizer' in net.prefixes_
        assert 'myoptimizer_' in net.cuda_dependent_attributes_
        assert 'myoptimizer' in net.prefixes_

        del net.myoptimizer_
        assert 'myoptimizer' not in net.prefixes_
        assert 'myoptimizer_' not in net.cuda_dependent_attributes_
        assert 'myoptimizer' not in net.prefixes_

    def test_custom_optimizer_virtual_params(self, net_cls, module_cls):
        # creating a custom optimizer should lead to its parameters being
        # virtual
        side_effects = []

        class MyNet(net_cls):
            def initialize_module(self):
                side_effects.append(True)
                return super().initialize_module()

            def initialize_optimizer(self):
                super().initialize_optimizer()
                self.myoptimizer_ = torch.optim.SGD(self.module_.parameters(), lr=1)
                return self

        net = MyNet(module_cls).initialize()

        # module initialized once
        assert len(side_effects) == 1

        net.set_params(optimizer__lr=123)
        # module is not re-initialized, since virtual parameter
        assert len(side_effects) == 1

        net.set_params(myoptimizer__lr=123)
        # module is not re-initialized, since virtual parameter
        assert len(side_effects) == 1

    def test_custom_optimizer_lr_is_associated_with_optimizer(
            self, net_cls, module_cls,
    ):
        # the 'lr' parameter belongs to the default optimizer, not any custom
        # optimizer
        class MyNet(net_cls):
            def initialize_optimizer(self):
                super().initialize_optimizer()
                self.myoptimizer_ = torch.optim.SGD(self.module_.parameters(), lr=1)
                return self

        net = MyNet(module_cls, lr=123).initialize()
        assert net.optimizer_.state_dict()['param_groups'][0]['lr'] == 123
        assert net.myoptimizer_.state_dict()['param_groups'][0]['lr'] == 1

        net.set_params(lr=456)
        assert net.optimizer_.state_dict()['param_groups'][0]['lr'] == 456
        assert net.myoptimizer_.state_dict()['param_groups'][0]['lr'] == 1

    def test_setattr_custom_module_no_duplicates(self, net_cls, module_cls):
        # the 'module' attribute is set twice but that shouldn't lead
        # to duplicates in prefixes_ or cuda_dependent_attributes_
        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.module_ = module_cls()  # same attribute name
                return self

        net = MyNet(module_cls).initialize()
        assert net.prefixes_.count('module') == 1
        assert net.cuda_dependent_attributes_.count('module_') == 1

    def test_setattr_in_initialize_non_torch_attribute(self, net_cls, module_cls):
        # attributes that are not torch modules or optimizers should
        # not be registered
        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.num = 123
                self.num_ = 123
                return self

        net = MyNet(module_cls)
        assert 'num' not in net.prefixes_
        assert 'num_' not in net.cuda_dependent_attributes_

    def test_setattr_does_not_modify_class_attribute(self, net_cls, module_cls):
        net = net_cls(module_cls)
        assert 'mymodule' not in net.prefixes_
        assert 'mymodule' not in net.cuda_dependent_attributes_

        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.mymodule_ = module_cls()
                return self

        net = MyNet(module_cls).initialize()
        assert 'mymodule' in net.prefixes_
        assert 'mymodule_' in net.cuda_dependent_attributes_

        assert 'mymodule' not in net_cls.prefixes_
        assert 'mymodule_' not in net_cls.cuda_dependent_attributes_

    @pytest.fixture
    def net_custom_module_cls(self, net_cls, module_cls):
        class MyNet(net_cls):
            """Net with custom attribute mymodule"""
            def __init__(self, *args, custom=module_cls, **kwargs):
                self.custom = custom
                super().__init__(*args, **kwargs)

            def initialize_module(self, *args, **kwargs):
                super().initialize_module(*args, **kwargs)

                params = self.get_params_for('custom')
                # pylint: disable=attribute-defined-outside-init
                self.custom_ = self.custom(**params)

                return self

        return MyNet

    def test_set_params_on_custom_module(self, net_custom_module_cls, module_cls):
        # set_params requires the prefixes_ attribute to be correctly
        # set, which is what is tested here
        net = net_custom_module_cls(module_cls, custom__hidden_units=77).initialize()
        hidden_units = net.custom_.state_dict()['sequential.3.weight'].shape[1]
        assert hidden_units == 77

        net.set_params(custom__hidden_units=99)
        hidden_units = net.custom_.state_dict()['sequential.3.weight'].shape[1]
        assert hidden_units == 99

    def test_save_load_state_dict_custom_module(
            self, net_custom_module_cls, module_cls, tmpdir):
        # test that we can store and load an arbitrary attribute like 'custom'
        net = net_custom_module_cls(module_cls).initialize()
        weights_before = net.custom_.state_dict()['sequential.3.weight']
        tmpdir_custom = str(tmpdir.mkdir('skorch').join('custom.pkl'))
        net.save_params(f_custom=tmpdir_custom)
        del net

        # initialize a new net, weights should differ
        net_new = net_custom_module_cls(module_cls).initialize()
        weights_new = net_new.custom_.state_dict()['sequential.3.weight']
        assert not (weights_before == weights_new).all()

        # after loading, weights should be the same again
        net_new.load_params(f_custom=tmpdir_custom)
        weights_loaded = net_new.custom_.state_dict()['sequential.3.weight']
        assert (weights_before == weights_loaded).all()

    def test_custom_module_params_passed_to_optimizer(
            self, net_custom_module_cls, module_cls):
        # custom module parameters should automatically be passed to the optimizer
        net = net_custom_module_cls(module_cls).initialize()
        optimizer = net.optimizer_

        module0 = net.module_
        module1 = net.custom_
        num_params_optimizer = len(optimizer.param_groups[0]['params'])
        num_params_expected = len(module0.state_dict()) + len(module1.state_dict())
        assert num_params_optimizer == num_params_expected

    def test_criterion_params_passed_to_optimizer_if_any(self, net_fit_criterion):
        # the parameters of the criterion should be passed to the optimizer if
        # there are any
        optimizer = net_fit_criterion.optimizer_

        num_params_module = len(net_fit_criterion.module_.state_dict())
        num_params_criterion = len(net_fit_criterion.criterion_.state_dict())
        num_params_optimizer = len(optimizer.param_groups[0]['params'])

        assert num_params_criterion > 0
        assert num_params_optimizer == num_params_module + num_params_criterion

    def test_set_params_on_custom_module_triggers_reinit_of_criterion_and_optimizer(
            self, net_custom_module_cls, module_cls,
    ):
        # When a custom module is re-initialized because of set_params, the
        # criterion and optimizer should also be re-initialized, as with a
        # normal module.
        init_side_effects = []  # record initialize calls

        class MyNet(net_custom_module_cls):
            """Records initialize_* calls"""
            def initialize_module(self):
                super().initialize_module()
                init_side_effects.append('module')
                return self

            def initialize_criterion(self):
                super().initialize_criterion()
                init_side_effects.append('criterion')
                return self

            def initialize_optimizer(self):
                super().initialize_optimizer()
                init_side_effects.append('optimizer')
                return self

        net = MyNet(module_cls).initialize()

        # just normal initialization behavior
        assert init_side_effects == ['module', 'criterion', 'optimizer']

        # still just normal behavior
        net.set_params(module__hidden_units=123)
        assert init_side_effects == ['module', 'criterion', 'optimizer'] * 2

        # setting custom module should also re-initialize
        net.set_params(custom__num_hidden=3)
        assert init_side_effects == ['module', 'criterion', 'optimizer'] * 3

        # setting normal and custom module should re-initialize, but only once
        net.set_params(module__num_hidden=1, custom__dropout=0.7)
        assert init_side_effects == ['module', 'criterion', 'optimizer'] * 4

    def test_set_params_on_custom_criterion_triggers_reinit_of_optimizer(
            self, net_cls, module_cls,
    ):
        # When a custom criterion is re-initialized because of set_params, the
        # optimizer should also be re-initialized, as with a normal criterion.
        init_side_effects = []  # record initialize calls

        class MyNet(net_cls):
            """Records initialize_* calls"""
            def __init__(self, *args, mycriterion, **kwargs):
                self.mycriterion = mycriterion
                super().__init__(*args, **kwargs)

            def initialize_module(self):
                super().initialize_module()
                init_side_effects.append('module')
                return self

            def initialize_criterion(self):
                super().initialize_criterion()
                params = self.get_params_for('mycriterion')
                self.mycriterion_ = self.mycriterion(**params)
                init_side_effects.append('criterion')
                return self

            def initialize_optimizer(self):
                super().initialize_optimizer()
                init_side_effects.append('optimizer')
                return self

        net = MyNet(module_cls, mycriterion=nn.NLLLoss).initialize()

        # just normal initialization behavior
        assert init_side_effects == ['module'] + ['criterion', 'optimizer']

        # still just normal behavior
        net.set_params(criterion__ignore_index=123)
        assert init_side_effects == ['module'] + ['criterion', 'optimizer'] * 2

        # setting custom module should also re-initialize
        net.set_params(mycriterion__ignore_index=456)
        assert init_side_effects == ['module'] + ['criterion', 'optimizer'] * 3

        # setting normal and custom module should re-initialize, but only once
        net.set_params(criterion__size_average=True, mycriterion__reduce=False)
        assert init_side_effects == ['module'] + ['criterion', 'optimizer'] * 4

    def test_set_params_on_custom_module_with_default_module_params_msg(
            self, net_cls, module_cls, capsys,
    ):
        # say we have module and module2, with module having some non-default
        # params, e.g. module__num_hidden=3; when setting params on module2,
        # that non-default value should not be given as a reason for
        # re-initialization.
        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.module2_ = module_cls()
                return self

        net = MyNet(module_cls, module__hidden_units=7).initialize()
        net.set_params(module2__num_hidden=3)

        msg = capsys.readouterr()[0]
        # msg should not be about hidden_units, since that wasn't changed, but
        # about num_hidden
        expected = ("Re-initializing module because the following parameters "
                    "were re-set: module2__num_hidden.")
        assert msg.startswith(expected)

    def test_set_params_on_custom_criterion_with_default_criterion_params_msg(
            self, net_cls, module_cls, capsys,
    ):
        # say we have criterion and criterion2, with criterion having some non-default
        # params, e.g. criterion__num_hidden=3; when setting params on criterion2,
        # that non-default value should not be given as a reason for
        # re-initialization.
        class MyNet(net_cls):
            def initialize_criterion(self):
                super().initialize_criterion()
                self.criterion2_ = module_cls()
                return self

        net = MyNet(module_cls, criterion__reduce=False).initialize()
        net.set_params(criterion2__num_hidden=3)

        msg = capsys.readouterr()[0]
        # msg should not be about hidden_units, since that wasn't changed, but
        # about num_hidden
        expected = ("Re-initializing criterion because the following parameters "
                    "were re-set: criterion2__num_hidden.")
        assert msg.startswith(expected)

    def test_modules_reinit_when_both_initialized_but_custom_module_changed(
            self, net_cls, module_cls,
    ):
        # When the default module and the custom module are already initialized,
        # initialize() should just leave them. However, when we change a
        # parameter on the custom module, both should be re-initialized.
        class MyNet(net_cls):
            def __init__(self, *args, module2, **kwargs):
                self.module2 = module2
                super().__init__(*args, **kwargs)

            def initialize_module(self):
                super().initialize_module()
                params = self.get_params_for('module2')
                is_init = isinstance(self.module2, nn.Module)

                if is_init and not params:
                    # no need to initialize
                    self.module2_ = self.module2
                    return

                if is_init:
                    module2 = type(self.module2)
                else:
                    module2 = self.module2
                self.module2_ = module2(**params)
                return self

        module = module_cls()
        module2 = module_cls()

        # all default params, hence no re-initilization
        net = MyNet(module=module, module2=module2).initialize()
        assert net.module_ is module
        assert net.module2_ is module2

        # module2 non default param, hence re-initilization
        net = MyNet(module=module, module2=module2, module2__num_hidden=3).initialize()
        assert net.module_ is module
        assert net.module2_ is not module2

    def test_criteria_reinit_when_both_initialized_but_custom_criterion_changed(
            self, net_cls, module_cls,
    ):
        # When the default criterion and the custom criterion are already initialized,
        # initialize() should just leave them. However, when we change a
        # parameter on the custom criterion, both should be re-initialized.
        class MyNet(net_cls):
            def __init__(self, *args, criterion2, **kwargs):
                self.criterion2 = criterion2
                super().__init__(*args, **kwargs)

            def initialize_criterion(self):
                super().initialize_criterion()
                params = self.get_params_for('criterion2')
                is_init = isinstance(self.criterion2, nn.Module)

                if is_init and not params:
                    # no need to initialize
                    self.criterion2_ = self.criterion2
                    return

                if is_init:
                    criterion2 = type(self.criterion2)
                else:
                    criterion2 = self.criterion2
                self.criterion2_ = criterion2(**params)
                return self

        criterion = module_cls()
        criterion2 = module_cls()

        # all default params, hence no re-initilization
        net = MyNet(module_cls, criterion=criterion, criterion2=criterion2).initialize()
        assert net.criterion_ is criterion
        assert net.criterion2_ is criterion2

        # criterion2 non default param, hence re-initilization
        net = MyNet(
            module_cls,
            criterion=criterion,
            criterion2=criterion2,
            criterion2__num_hidden=3,
        ).initialize()
        assert net.criterion_ is criterion
        assert net.criterion2_ is not criterion2

    def test_custom_module_is_init_when_default_module_already_is(
            self, net_cls, module_cls,
    ):
        # Assume that the module is already initialized, which is something we
        # allow, but the custom module isn't. After calling initialize(), the
        # custom module should be initialized and not skipped just because the
        # default module already was initialized.
        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.module2_ = module_cls()
                return self

        module = module_cls()
        net = MyNet(module=module).initialize()  # module already initialized

        assert net.module_ is module  # normal module_ not changed
        assert hasattr(net, 'module2_')  # there is a module2_

    def test_custom_criterion_is_init_when_default_criterion_already_is(
            self, net_cls, module_cls,
    ):
        # Assume that the criterion is already initialized, which is something we
        # allow, but the custom criterion isn't. After calling initialize(), the
        # custom criterion should be initialized and not skipped just because the
        # default criterion already was initialized.
        class MyNet(net_cls):
            def initialize_criterion(self):
                super().initialize_criterion()
                self.criterion2_ = module_cls()
                return self

        criterion = module_cls()
        # criterion already initialized
        net = MyNet(module_cls, criterion=criterion).initialize()

        assert net.criterion_ is criterion  # normal criterion_ not changed
        assert hasattr(net, 'criterion2_')  # there is a criterion2_

    def test_setting_custom_module_outside_initialize_raises(self, net_cls, module_cls):
        from skorch.exceptions import SkorchAttributeError

        # all modules should be set within an initialize method
        class MyNet(net_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.foo_ = module_cls()

        msg = ("Trying to set torch compoment 'foo_' outside of an initialize method. "
               "Consider defining it inside 'initialize_module'")
        with pytest.raises(SkorchAttributeError, match=msg):
            MyNet(module_cls)

    def test_setting_custom_optimizer_outside_initialize_raises(
            self, net_cls, module_cls
    ):
        from skorch.exceptions import SkorchAttributeError

        # all optimzers should be set within an initialize method
        class MyNet(net_cls):
            def initialize(self):
                super().initialize()
                self.opti = torch.optim.Adam(self.module_.parameters())
                return self

        msg = ("Trying to set torch compoment 'opti' outside of an initialize method. "
               "Consider defining it inside 'initialize_optimizer'")
        with pytest.raises(SkorchAttributeError, match=msg):
            MyNet(module_cls).initialize()

    def test_setting_custom_module_without_trailing_underscore_raises(
            self, net_cls, module_cls,
    ):
        from skorch.exceptions import SkorchAttributeError

        # all initialized modules should end on an underscore
        class MyNet(net_cls):
            def initialize_module(self):
                super().initialize_module()
                self.mymodule = module_cls()
                return self

        msg = ("Names of initialized modules or optimizers should end "
               "with an underscore (e.g. 'mymodule_')")
        with pytest.raises(SkorchAttributeError, match=re.escape(msg)):
            MyNet(module_cls).initialize()

    def test_moving_custom_modules_to_device(self, net_cls):
        # testing that custom modules and criteria are moved to the indicated
        # device, not just the normal module/criterion; we override .to(device)
        # here to be able to test this even without GPU

        device_side_effects = []  # record module name and device

        class MyModule(nn.Module):
            """Custom module that records .to calls"""
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.lin = nn.Linear(5, 5)  # module needs parameters

            def to(self, device):
                device_side_effects.append((self.name, device))
                return self

        class MyNet(net_cls):
            """Net with custom mymodule and mycriterion"""
            def __init__(self, *args, mymodule, mycriterion, **kwargs):
                self.mymodule = mymodule
                self.mycriterion = mycriterion
                super().__init__(*args, **kwargs)

            def initialize_module(self):
                super().initialize_module()
                params = self.get_params_for('mymodule')
                self.mymodule_ = MyModule(**params)
                return self

            def initialize_criterion(self):
                super().initialize_criterion()
                params = self.get_params_for('mycriterion')
                self.mycriterion_ = MyModule(**params)
                return self

        MyNet(
            module=MyModule,
            module__name='module-normal',
            mymodule=MyModule,
            mymodule__name='module-custom',

            criterion=MyModule,
            criterion__name='criterion-normal',
            mycriterion=MyModule,
            mycriterion__name='criterion-custom',

            device='foo',
        ).initialize()

        expected = [('module-normal', 'foo'), ('module-custom', 'foo'),
                    ('criterion-normal', 'foo'), ('criterion-custom', 'foo')]
        assert device_side_effects == expected

    def test_set_params_on_custom_module_preserves_its_device(self, net_cls):
        # when a custom module or criterion is re-created because of set_params,
        # it should be moved to the indicated device
        class MyNet(net_cls):
            """Net with custom module and criterion"""
            def __init__(self, *args, mymodule, mycriterion, **kwargs):
                self.mymodule = mymodule
                self.mycriterion = mycriterion
                super().__init__(*args, **kwargs)

            def initialize_module(self):
                super().initialize_module()
                params = self.get_params_for('mymodule')
                self.mymodule_ = self.mymodule(**params)
                return self

            def initialize_criterion(self):
                super().initialize_criterion()
                params = self.get_params_for('mycriterion')
                self.mycriterion_ = self.mycriterion(**params)
                return self

        class MyModule(nn.Module):
            """Custom module to test device even without GPU"""
            def __init__(self, x=1):
                super().__init__()
                self.lin = nn.Linear(x, 1)  # modules need parameters
                self.device = 'cpu'

            def to(self, device):
                self.device = device
                return self

        # first normal CPU
        net = MyNet(
            module=MyModule,
            mymodule=MyModule,
            criterion=MyModule,
            mycriterion=MyModule,
        ).initialize()
        assert net.mymodule_.device == 'cpu'
        assert net.mycriterion_.device == 'cpu'

        # now try other device
        net = MyNet(
            module=MyModule,
            mymodule=MyModule,
            device='foo',
            criterion=MyModule,
            mycriterion=MyModule,
        ).initialize()
        assert net.mymodule_.device == 'foo'
        assert net.mycriterion_.device == 'foo'

        net.set_params(mymodule__x=3)
        assert net.mymodule_.device == 'foo'
        assert net.mycriterion_.device == 'foo'

    def test_custom_modules_and_criteria_training_mode_set_correctly(
            self, net_cls, module_cls, data,
    ):
        # custom modules and criteria should be set to training/eval mode
        # correctly depending on the stage of training/validation/inference
        from skorch.callbacks import Callback

        class MyNet(net_cls):
            """Net with custom mymodule and mycriterion"""
            def initialize_module(self):
                super().initialize_module()
                self.mymodule_ = module_cls()
                return self

            def initialize_criterion(self):
                super().initialize_criterion()
                self.mycriterion_ = module_cls()
                return self

            def evaluation_step(self, batch, training=False):
                y_pred = super().evaluation_step(batch, training=training)
                assert_net_training_mode(self, training=training)
                return y_pred

        def assert_net_training_mode(net, training=True):
            if training:
                check = lambda module: module.training is True
            else:
                check = lambda module: module.training is False
            assert check(net.module_)
            assert check(net.mymodule_)
            assert check(net.criterion_)
            assert check(net.mycriterion_)

        class CheckTrainingCallback(Callback):
            def on_batch_end(self, net, batch, training, **kwargs):
                assert_net_training_mode(net, training=training)

        X, y = data
        net = MyNet(module_cls, callbacks=[CheckTrainingCallback], max_epochs=1)
        net.fit(X, y)
        net.predict(X)

    def test_custom_optimizer_performs_updates(self, net_cls, module_cls, data):
        # make sure that updates are actually performed by a custom optimizer
        from skorch.utils import to_tensor

        # custom optimizers should actually perform updates
        # pylint: disable=attribute-defined-outside-init
        class MyNet(net_cls):
            """A net with 2 modules with their respective optimizers"""
            def initialize_module(self):
                super().initialize_module()
                self.module2_ = module_cls()
                return self

            def initialize_optimizer(self):
                self.optimizer_ = self.optimizer(self.module_.parameters(), self.lr)
                self.optimizer2_ = self.optimizer(self.module2_.parameters(), self.lr)
                return self

            def infer(self, x, **fit_params):
                # prediction is just mean of the two modules
                x = to_tensor(x, device=self.device)
                return 0.5 * (self.module_(x) + self.module2_(x))

        net = MyNet(module_cls, max_epochs=1, lr=0.5).initialize()
        # params1_before = [copy.deepcopy(p) for p in net.module_.parameters()]
        params1_before = copy.deepcopy(list(net.module_.parameters()))
        params2_before = copy.deepcopy(list(net.module2_.parameters()))

        net.partial_fit(*data)
        params1_after = list(net.module_.parameters())
        params2_after = list(net.module2_.parameters())

        assert not any(
            (p_b == p_a).all() for p_b, p_a in zip(params1_before, params1_after))
        assert not any(
            (p_b == p_a).all() for p_b, p_a in zip(params2_before, params2_after))

    def test_optimizer_initialized_after_module_moved_to_device(self, net_cls):
        # it is recommended to initialize the optimizer with the module params
        # _after_ the module has been moved to its device, see:
        # https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/6

        side_effects = []  # record module name and device

        class MyModule(nn.Module):
            """Custom module that records .to calls"""
            def __init__(self, x=1):
                super().__init__()
                self.lin = nn.Linear(x, 1)  # module needs parameters

            def to(self, device):
                side_effects.append('moved-to-device')
                return self

        class MyOptimizer(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                side_effects.append('optimizer-init')

        net = net_cls(
            module=MyModule,
            criterion=MyModule,
            optimizer=MyOptimizer,
        ).initialize()

        # first move module and criterion to device, then initialize optimizer
        expected = ['moved-to-device', 'moved-to-device', 'optimizer-init']
        assert side_effects == expected

        net.set_params(module__x=2)
        # after set_params on module, re-initialization and moving of device
        # should happen again, with the same order as before
        expected = ['moved-to-device', 'moved-to-device', 'optimizer-init'] * 2
        assert side_effects == expected

    @pytest.mark.parametrize("needs_y, train_split, raises", [
        (False, None, ExitStack()),  # ExitStack = does not raise
        (True, None, ExitStack()),
        (False, "default", ExitStack()),  # Default parameters for NeuralNet
        (True, "default", ExitStack()),  # Default parameters for NeuralNet
        (False, lambda x: (x, x), ExitStack()),  # Earlier this was not allowed
        (True, lambda x, y: (x, x), ExitStack()),  # Works for custom split
        (True, lambda x: (x, x), pytest.raises(TypeError)),  # Raises an error
    ])
    def test_passes_y_to_train_split_when_not_none(
            self, needs_y, train_split, raises):
        from skorch.net import NeuralNet
        from skorch.toy import MLPModule

        # By default, `train_split=CVSplit(5)` in the `NeuralNet` definition
        kwargs = {} if train_split == 'default' else {
            'train_split': train_split}

        # Dummy loss that ignores y_true
        class UnsupervisedLoss(torch.nn.NLLLoss):
            def forward(self, y_pred, _):
                return y_pred.mean()

        # Generate the dummy dataset
        n_samples, n_features = 128, 10
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.random.binomial(n=1, p=0.5, size=n_samples) if needs_y else None

        # The `NeuralNetClassifier` or `NeuralNetRegressor` always require `y`
        # Only `NeuralNet` can transfer `y=None` to `train_split` method.
        net = NeuralNet(
            MLPModule,  # Any model, it's not important here
            module__input_units=n_features,
            max_epochs=2,  # Run train loop twice to detect possible errors
            criterion=UnsupervisedLoss,
            **kwargs,
        )

        # Check if the code should fail or not
        with raises:
            net.fit(X, y)

    def test_predict_nonlinearity_called_with_predict(
            self, net_cls, module_cls, data):
        side_effect = []
        def nonlin(X):
            side_effect.append(X)
            return np.zeros_like(X)

        X, y = data[0][:200], data[1][:200]
        net = net_cls(
            module_cls, max_epochs=1, predict_nonlinearity=nonlin).initialize()

        # don't want callbacks to trigger side effects
        net.callbacks_ = []
        net.partial_fit(X, y)
        assert not side_effect

        # 2 calls, since batch size == 128 and n == 200
        y_proba = net.predict(X)
        assert len(side_effect) == 2
        assert side_effect[0].shape == (128, 2)
        assert side_effect[1].shape == (72, 2)
        assert (y_proba == 0).all()

        net.predict(X)
        assert len(side_effect) == 4

    def test_predict_nonlinearity_called_with_predict_proba(
            self, net_cls, module_cls, data):
        side_effect = []
        def nonlin(X):
            side_effect.append(X)
            return np.zeros_like(X)

        X, y = data[0][:200], data[1][:200]
        net = net_cls(
            module_cls, max_epochs=1, predict_nonlinearity=nonlin).initialize()

        net.callbacks_ = []
        # don't want callbacks to trigger side effects
        net.partial_fit(X, y)
        assert not side_effect

        # 2 calls, since batch size == 128 and n == 200
        y_proba = net.predict_proba(X)
        assert len(side_effect) == 2
        assert side_effect[0].shape == (128, 2)
        assert side_effect[1].shape == (72, 2)
        assert np.allclose(y_proba, 0)

        net.predict_proba(X)
        assert len(side_effect) == 4

    def test_predict_nonlinearity_none(
            self, net_cls, module_cls, data):
        # even though we have CrossEntropyLoss, we don't want the
        # output from predict_proba to be modified, thus we set
        # predict_nonlinearity to None
        X = data[0][:200]
        net = net_cls(
            module_cls,
            max_epochs=1,
            criterion=nn.CrossEntropyLoss,
            predict_nonlinearity=None,
        ).initialize()

        rv = np.random.random((20, 5))
        net.forward_iter = (
            lambda *args, **kwargs: (torch.as_tensor(rv) for _ in range(2)))

        # 2 batches, mock return value has shape 20,5 thus y_proba has
        # shape 40,5
        y_proba = net.predict_proba(X)
        assert y_proba.shape == (40, 5)
        assert np.allclose(y_proba[:20], rv)
        assert np.allclose(y_proba[20:], rv)

    def test_predict_nonlinearity_type_error(self, net_cls, module_cls):
        # if predict_nonlinearity is not callable, raise a TypeError
        net = net_cls(module_cls, predict_nonlinearity=123).initialize()

        msg = "predict_nonlinearity has to be a callable, 'auto' or None"
        with pytest.raises(TypeError, match=msg):
            net.predict(np.zeros((3, 3)))

        with pytest.raises(TypeError, match=msg):
            net.predict_proba(np.zeros((3, 3)))

    def test_customize_net_with_custom_dataset_that_returns_3_values(self, data):
        # Test if it's possible to easily customize NeuralNet to work
        # with Datasets that don't return 2 values. This way, a user
        # can more easily customize the net and use his or her own
        # datasets.
        from skorch import NeuralNet
        from skorch.utils import to_tensor

        class MyDataset(torch.utils.data.Dataset):
            """Returns 3 elements instead of 2"""
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __getitem__(self, i):
                x = self.X[i]
                if self.y is None:
                    return x[:5], x[5:]
                y = self.y[i]
                return x[:5], x[5:], y

            def __len__(self):
                return len(self.X)

        class MyModule(nn.Module):
            """Module that takes 2 inputs"""
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(20, 2)

            def forward(self, x0, x1):
                x = torch.cat((x0, x1), axis=1)
                return self.lin(x)

        class MyNet(NeuralNet):
            """Override train_step_single and validation_step"""
            def train_step_single(self, batch, **fit_params):
                self._set_training(True)
                x0, x1, yi = batch
                x0, x1, yi = to_tensor((x0, x1, yi), device=self.device)
                y_pred = self.module_(x0, x1)
                loss = self.criterion_(y_pred, yi)
                loss.backward()
                return {'loss': loss, 'y_pred': y_pred}

            def validation_step(self, batch, **fit_params):
                self._set_training(False)
                x0, x1, yi = batch
                x0, x1, yi = to_tensor((x0, x1, yi), device=self.device)
                y_pred = self.module_(x0, x1)
                loss = self.criterion_(y_pred, yi)
                return {'loss': loss, 'y_pred': y_pred}

            def evaluation_step(self, batch, training=False):
                self.check_is_fitted()
                x0, x1 = batch
                x0, x1 = to_tensor((x0, x1), device=self.device)
                with torch.set_grad_enabled(training):
                    self._set_training(training)
                    return self.module_(x0, x1)

        net = MyNet(
            MyModule,
            lr=0.1,
            dataset=MyDataset,
            criterion=nn.CrossEntropyLoss,
        )
        X, y = data[0][:100], data[1][:100]
        net.fit(X, y)

        # net learns
        assert net.history[-1, 'train_loss'] < 0.75 * net.history[0, 'train_loss']

        y_pred = net.predict(X)
        assert y_pred.shape == (100, 2)


class TestNetSparseInput:
    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture
    def net(self, net_cls, module_cls):
        return net_cls(module_cls, lr=0.02, max_epochs=20)

    @pytest.fixture
    def model(self, net):
        return Pipeline([
            # TfidfVectorizer returns a scipy sparse CSR matrix
            ('tfidf', TfidfVectorizer(max_features=20, dtype=np.float32)),
            ('net', net),
        ])

    @pytest.fixture(scope='module')
    def X(self):
        with open(__file__, 'r') as f:
            lines = f.readlines()
        return np.asarray(lines)

    @pytest.fixture(scope='module')
    def y(self, X):
        return np.array(
            [1 if (' def ' in x) or (' assert ' in x) else 0 for x in X])

    def test_fit_sparse_csr_learns(self, model, X, y):
        model.fit(X, y)
        net = model.steps[-1][1]
        score_start = net.history[0]['train_loss']
        score_end = net.history[-1]['train_loss']

        assert score_start > 1.25 * score_end

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_fit_sparse_csr_learns_cuda(self, model, X, y):
        model.set_params(net__device='cuda')
        model.fit(X, y)
        net = model.steps[-1][1]
        score_start = net.history[0]['train_loss']
        score_end = net.history[-1]['train_loss']

        assert score_start > 1.25 * score_end
