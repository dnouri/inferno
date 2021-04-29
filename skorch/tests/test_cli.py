"""Test for cli.py"""

from math import cos
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import RReLU


fire_installed = True
try:
    import fire  # pylint: disable=unused-import
except ImportError:
    fire_installed = False


@pytest.mark.skipif(not fire_installed, reason='fire libarary not installed')
class TestCli:
    @pytest.fixture
    def resolve_dotted_name(self):
        from skorch.cli import _resolve_dotted_name
        return _resolve_dotted_name

    @pytest.mark.parametrize('name, expected', [
        (0, 0),
        (1.23, 1.23),
        ('foo', 'foo'),
        ('math.cos', cos),
        ('torch.nn', nn),
        ('torch.nn.ReLU', nn.ReLU),
    ])
    def test_resolve_dotted_name(self, resolve_dotted_name, name, expected):
        result = resolve_dotted_name(name)
        assert result == expected

    def test_resolve_dotted_name_instantiated(self, resolve_dotted_name):
        result = resolve_dotted_name('torch.nn.RReLU(0.123, upper=0.456)')
        assert isinstance(result, RReLU)
        assert np.isclose(result.lower, 0.123)
        assert np.isclose(result.upper, 0.456)

    @pytest.fixture
    def parse_net_kwargs(self):
        from skorch.cli import parse_net_kwargs
        return parse_net_kwargs

    def test_parse_net_kwargs(self, parse_net_kwargs):
        kwargs = {
            'lr': 0.05,
            'max_epochs': 5,
            'module__num_units': 10,
            'module__nonlin': 'torch.nn.RReLU(0.123, upper=0.456)',
        }
        parsed_kwargs = parse_net_kwargs(kwargs)

        assert len(parsed_kwargs) == 4
        assert np.isclose(parsed_kwargs['lr'], 0.05)
        assert parsed_kwargs['max_epochs'] == 5
        assert parsed_kwargs['module__num_units'] == 10
        assert isinstance(parsed_kwargs['module__nonlin'], RReLU)
        assert np.isclose(parsed_kwargs['module__nonlin'].lower, 0.123)
        assert np.isclose(parsed_kwargs['module__nonlin'].upper, 0.456)

    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def net(self, net_cls, classifier_module):
        return net_cls(classifier_module)

    @pytest.fixture
    def pipe(self, net):
        return Pipeline([
            ('features', FeatureUnion([
                ('scale', MinMaxScaler()),
            ])),
            ('net', net),
        ])

    @pytest.fixture
    def yield_estimators(self):
        from skorch.cli import _yield_estimators
        return _yield_estimators

    def test_yield_estimators_net(self, yield_estimators, net):
        result = list(yield_estimators(net))

        assert result[0][0] == ''
        assert result[0][1] is net
        assert result[1][0] == 'module'
        assert result[1][1] is net.module

    def test_yield_estimators_pipe(self, yield_estimators, pipe):
        result = list(yield_estimators(pipe))
        scaler = pipe.named_steps['features'].transformer_list[0][1]
        net = pipe.named_steps['net']
        module = net.module

        assert result[0][0] == 'features__scale'
        assert result[0][1] is scaler
        assert result[1][0] == 'net'
        assert result[1][1] is net
        assert result[2][0] == 'net__module'
        assert result[2][1] is module

    @pytest.fixture
    def substitute_default(self):
        from skorch.cli import _substitute_default
        return _substitute_default

    @pytest.mark.parametrize('s, new_value, expected', [
        ('', '', ''),
        ('', 'foo', ''),
        ('bar', 'foo', 'bar'),
        ('int (default=128)', '', 'int (default=)'),
        ('int (default=128)', None, 'int (default=128)'),
        ('int (default=128)', '""', 'int (default="")'),
        ('int (default=128)', '128', 'int (default=128)'),
        ('int (default=128)', '256', 'int (default=256)'),
        ('int (default=128)', 256, 'int (default=256)'),
        ('with_parens (default=(1, 2))', (3, 4), 'with_parens (default=(3, 4))'),
        ('int (default =128)', '256', 'int (default =256)'),
        ('int (default= 128)', '256', 'int (default= 256)'),
        ('int (default = 128)', '256', 'int (default = 256)'),
        (
            'nonlin (default = ReLU())',
            nn.Hardtanh(1, 2),
            'nonlin (default = {})'.format(nn.Hardtanh(1, 2))
        ),
        (
            # from sklearn MinMaxScaler
            'tuple (min, max), default=(0, 1)',
            (-1, 1),
            'tuple (min, max), default=(-1, 1)'
        ),
        (
            # from sklearn MinMaxScaler
            'boolean, optional, default True',
            False,
            'boolean, optional, default False'
        ),
        (
            # from sklearn Normalizer
            "'l1', 'l2', or 'max', optional ('l2' by default)",
            'l1',
            "'l1', 'l2', or 'max', optional ('l1' by default)"
        ),
        (
            # same but double ticks
            '"l1", "l2", or "max", optional ("l2" by default)',
            'l1',
            '"l1", "l2", or "max", optional ("l1" by default)'
        ),
        (
            # same but no ticks
            "l1, l2, or max, optional (l2 by default)",
            'l1',
            "l1, l2, or max, optional (l1 by default)"
        ),
        (
            "tuple, optional ((1, 1) by default)",
            (2, 2),
            "tuple, optional ((2, 2) by default)"
        ),
        (
            "nonlin (ReLU() by default)",
            nn.Tanh(),
            "nonlin (Tanh() by default)"
        ),
    ])
    def test_replace_default(self, substitute_default, s, new_value, expected):
        result = substitute_default(s, new_value)
        assert result == expected

    @pytest.fixture
    def print_help(self):
        from skorch.cli import print_help
        return print_help

    def test_print_help_net(self, print_help, net, capsys):
        print_help(net)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<NeuralNetClassifier> options',
            '--module : torch module (class or instance)',
            '--batch_size : int (default=128)',
            '<MLPModule> options',
            '--module__hidden_units : int (default=10)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    def test_print_help_net_custom_defaults(self, print_help, net, capsys):
        defaults = {'batch_size': 256, 'module__hidden_units': 55}
        print_help(net, defaults)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<NeuralNetClassifier> options',
            '--module : torch module (class or instance)',
            '--batch_size : int (default=256)',
            '<MLPModule> options',
            '--module__hidden_units : int (default=55)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    def test_print_help_pipeline(self, print_help, pipe, capsys):
        print_help(pipe)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<MinMaxScaler> options',
            '--features__scale__feature_range',
            '<NeuralNetClassifier> options',
            '--net__module : torch module (class or instance)',
            '--net__batch_size : int (default=128)',
            '<MLPModule> options',
            '--net__module__hidden_units : int (default=10)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    def test_print_help_pipeline_custom_defaults(
            self, print_help, pipe, capsys):
        defaults = {'net__batch_size': 256, 'net__module__hidden_units': 55}
        print_help(pipe, defaults=defaults)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<MinMaxScaler> options',
            '--features__scale__feature_range',
            '<NeuralNetClassifier> options',
            '--net__module : torch module (class or instance)',
            '--net__batch_size : int (default=256)',
            '<MLPModule> options',
            '--net__module__hidden_units : int (default=55)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    @pytest.fixture
    def clf_sklearn(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    @pytest.fixture
    def pipe_sklearn(self, clf_sklearn):
        return Pipeline([
            ('features', FeatureUnion([
                ('scale', MinMaxScaler()),
            ])),
            ('clf', clf_sklearn),
        ])

    def test_print_help_sklearn_estimator(self, print_help, clf_sklearn, capsys):
        # Should also work with non-skorch sklearn estimator;
        # need to assert that count==1 since there was a bug in my
        # first implementation that resulted in the help for the final
        # estimator appearing twice.
        print_help(clf_sklearn)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '--fit_intercept',
            '--copy_X',
            '--normalize',
        ]
        for snippet in expected_snippets:
            assert snippet in out
            assert out.count(snippet) == 1

    def test_print_help_sklearn_pipeline(self, print_help, pipe_sklearn, capsys):
        # Should also work with non-skorch sklearn pipelines;
        # need to assert that count==1 since there was a bug in my
        # first implementation that resulted in the help for the final
        # estimator appearing twice.
        print_help(pipe_sklearn)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<MinMaxScaler> options',
            '--features__scale__feature_range',
            '--clf__fit_intercept',
            '--clf__normalize',
        ]
        for snippet in expected_snippets:
            assert snippet in out
            assert out.count(snippet) == 1

    @pytest.fixture
    def parse_args(self):
        from skorch.cli import parse_args
        return parse_args

    @pytest.fixture
    def estimator(self, net_cls):
        mock = Mock(net_cls)
        return mock

    def test_parse_args_help(self, parse_args, estimator):
        with patch('skorch.cli.sys.exit') as exit_:
            with patch('skorch.cli.print_help') as help_:
                parsed = parse_args({'help': True, 'foo': 'bar'})
                parsed(estimator)

        assert estimator.set_params.call_count == 0  # kwargs and defaults
        assert help_.call_count == 1
        assert exit_.call_count == 1

    def test_parse_args_run(self, parse_args, estimator):
        kwargs = {'foo': 'bar', 'baz': 'math.cos'}
        with patch('skorch.cli.sys.exit') as exit_:
            with patch('skorch.cli.print_help') as help_:
                parsed = parse_args(kwargs)
                parsed(estimator)

        assert estimator.set_params.call_count == 2  # defaults and kwargs

        defaults_set_params = estimator.set_params.call_args_list[0][1]
        assert not defaults_set_params  # no defaults specified

        kwargs_set_params = estimator.set_params.call_args_list[1][1]
        assert kwargs_set_params['foo'] == 'bar'
        # pylint: disable=comparison-with-callable
        assert kwargs_set_params['baz'] == cos

        assert help_.call_count == 0
        assert exit_.call_count == 0

    def test_parse_args_net_custom_defaults(self, parse_args, net):
        defaults = {'batch_size': 256, 'module__hidden_units': 55}
        kwargs = {'batch_size': 123, 'module__nonlin': nn.Hardtanh(1, 2)}
        parsed = parse_args(kwargs, defaults)
        net = parsed(net)

        # cmd line args have precedence over defaults
        assert net.batch_size == 123
        assert net.module__hidden_units == 55
        assert isinstance(net.module__nonlin, nn.Hardtanh)
        assert net.module__nonlin.min_val == 1
        assert net.module__nonlin.max_val == 2

    def test_parse_args_pipe_custom_defaults(self, parse_args, pipe):
        defaults = {'net__batch_size': 256, 'net__module__hidden_units': 55}
        kwargs = {'net__batch_size': 123, 'net__module__nonlin': nn.Hardtanh(1, 2)}
        parsed = parse_args(kwargs, defaults)
        pipe = parsed(pipe)
        net = pipe.steps[-1][1]

        # cmd line args have precedence over defaults
        assert net.batch_size == 123
        assert net.module__hidden_units == 55
        assert isinstance(net.module__nonlin, nn.Hardtanh)
        assert net.module__nonlin.min_val == 1
        assert net.module__nonlin.max_val == 2

    def test_parse_args_sklearn_pipe_custom_defaults(self, parse_args, pipe_sklearn):
        defaults = {'features__scale__copy': 123, 'clf__fit_intercept': 456}
        kwargs = {'features__scale__copy': 555, 'clf__normalize': 789}
        parsed = parse_args(kwargs, defaults)
        pipe = parsed(pipe_sklearn)
        scaler = pipe.steps[0][1].transformer_list[0][1]
        clf = pipe.steps[-1][1]

        # cmd line args have precedence over defaults
        assert scaler.copy == 555
        assert clf.fit_intercept == 456
        assert clf.normalize == 789
