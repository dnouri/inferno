"""TODO"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from skorch.utils import to_numpy


class TestGradientNormClipping:
    @pytest.fixture
    def grad_clip_cls_and_mock(self):
        with patch('skorch.callbacks.regularization.clip_grad_norm_') as cgn:
            from skorch.callbacks import GradientNormClipping
            yield GradientNormClipping, cgn

    @pytest.fixture
    def net_mock(self):
        return Mock(amp_enabled=False)

    def test_parameters_passed_correctly_to_torch_cgn(
            self, grad_clip_cls_and_mock, net_mock):
        grad_norm_clip_cls, cgn = grad_clip_cls_and_mock

        clipping = grad_norm_clip_cls(
            gradient_clip_value=55, gradient_clip_norm_type=99)
        named_parameters = [('p1', 1), ('p2', 2), ('p3', 3)]
        parameter_values = [p for _, p in named_parameters]
        clipping.on_grad_computed(net_mock, named_parameters=named_parameters)

        # Clip norm must receive values, not (name, value) pairs.
        assert list(cgn.call_args_list[0][0][0]) == parameter_values
        assert cgn.call_args_list[0][1]['max_norm'] == 55
        assert cgn.call_args_list[0][1]['norm_type'] == 99

    def test_no_parameter_updates_when_norm_0(
            self, classifier_module, classifier_data):
        from copy import deepcopy
        from skorch import NeuralNetClassifier
        from skorch.callbacks import GradientNormClipping

        net = NeuralNetClassifier(
            classifier_module,
            callbacks=[('grad_norm', GradientNormClipping(0))],
            train_split=None,
            warm_start=True,
            max_epochs=1,
        )
        net.initialize()

        params_before = deepcopy(list(net.module_.parameters()))
        net.fit(*classifier_data)
        params_after = net.module_.parameters()
        for p0, p1 in zip(params_before, params_after):
            p0, p1 = to_numpy(p0), to_numpy(p1)
            assert np.allclose(p0, p1)

    def test_with_amp_enabled(self):
        # TODO
        pass
