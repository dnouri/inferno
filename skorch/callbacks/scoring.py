from functools import partial
import warnings

import numpy as np
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score

from skorch.utils import data_from_dataset
from skorch.utils import to_numpy
from skorch.callbacks import Callback


__all__ = ['BatchScoring', 'EpochScoring']


class ScoringBase(Callback):
    """Base class for scoring.

    Subclass and implement an ``on_*`` method before using.
    """
    def __init__(
            self,
            scoring,
            lower_is_better=True,
            on_train=False,
            name=None,
            target_extractor=to_numpy,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.target_extractor = target_extractor

    def _get_name(self):
        if self.name is not None:
            return self.name
        if self.scoring is None:
            return 'score'
        if isinstance(self.scoring, str):
            return self.scoring
        if isinstance(self.scoring, partial):
            return self.scoring.func.__name__
        return self.scoring.__name__

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        self.name_ = self._get_name()
        return self

    def _scoring(self, net, X_test, y_test):
        """Resolve scoring and apply it to data."""
        scorer = check_scoring(net, self.scoring)
        scores = _score(
            estimator=net,
            X_test=X_test,
            y_test=y_test,
            scorer=scorer,
            is_multimetric=False,
        )
        return scores

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_


class BatchScoring(ScoringBase):
    """Callback that performs generic scoring on batches.

    This callback determines the score after each batch and stores it
    in the net's history in the column given by ``name``. At the end
    of the epoch, the average of the scores are determined and also
    stored in the history. Furthermore, it is determined whether this
    average score is the best score yet and that information is also
    stored in the history.

    In contrast to ``EpochScoring``, this callback determines the
    score for each batch and then averages the score at the end of the
    epoch. This can be disadvantageous for some scores if the batch
    size is small -- e.g. area under the ROC will return incorrect
    scores in this case. Therefore, it is recommnded to use
    ``EpochScoring`` unless you really need the scores for each batch.

    Parameters
    ----------
    scoring : None, str, or callable
      If None, use the ``score`` method of the model. If str, it should
      be a valid sklearn metric (e.g. "f1_score", "accuracy_score"). If
      a callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the ``scoring``
      parameter in sklearn's ``GridSearchCV`` et al.

    lower_is_better : bool (default=True)
      Whether lower (e.g. log loss) or higher (e.g. accuracy) scores
      are better

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    """
    # pylint: disable=unused-argument,arguments-differ
    def on_batch_end(self, net, X, y, training, **kwargs):
        if training != self.on_train:
            return

        y = self.target_extractor(y)
        try:
            score = self._scoring(net, X, y)
            net.history.record_batch(self.name_, score)
        except KeyError:
            pass

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name_]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', is_best)


class EpochScoring(ScoringBase):
    """Callback that performs generic scoring on predictions.

    At the end of each epoch, this callback makes a prediction on
    train or validation data, determines the score for that prediction
    and whether it is the best yet, and stores the result in the net's
    history.

    In case you already computed a score value for each batch you
    can omit the score computation step by return the value from
    the history. For example:

        >>> def my_score(net, X=None, y=None):
        ...     losses = net.history[-1, 'batches', :, 'my_score']
        ...     batch_sizes = net.history[-1, 'batches', :, 'valid_batch_size']
        ...     return np.average(losses, weights=batch_sizes)
        >>> net = MyNet(callbacks=[
        ...     ('my_score', Scoring(my_score, name='my_score'))

    If you fit with a custom dataset, this callback might be skipped,
    since it must explicitely find X and y to pass those values to
    sklearn's scoring functions. To circumvent this, your dataset
    should have an 'X' and a 'y' attribute that contain the input and
    target data, respectively.

    Parameters
    ----------
    scoring : None, str, or callable (default=None)
      If None, use the ``score`` method of the model. If str, it
      should be a valid sklearn scorer (e.g. "f1", "accuracy"). If a
      callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the
      ``scoring`` parameter in sklearn's ``GridSearchCV`` et al.

    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.

    on_train : bool (default=False)
      Whether this should be called during train or validation data.

    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    """
    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(
            self,
            net,
            dataset_train,
            dataset_valid,
            **kwargs):
        dataset = dataset_train if self.on_train else dataset_valid
        try:
            X_test, y_test = data_from_dataset(dataset)
        except AttributeError:
            X_test, y_test = None, None
            warnings.warn(
                "EpochScoring cannot access X and y from the dataset. This "
                "typically happens when you fit the net with your own "
                "dataset. In that case, EpochScoring does not work.")

        if X_test is None:
            return

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)

        history = net.history
        current_score = self._scoring(net, X_test, y_test)
        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', is_best)
        if is_best:
            self.best_score_ = current_score
