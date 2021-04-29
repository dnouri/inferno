"""Contains skorch-specific exceptions and warnings."""


class SkorchException(BaseException):
    """Base skorch exception."""


class NotInitializedError(SkorchException):
    """Module is not initialized, please call the ``.initialize``
    method or train the model by calling ``.fit(...)``.

    """


class SkorchAttributeError(SkorchException):
    """An attribute was set incorrectly on a skorch net."""


class SkorchWarning(UserWarning):
    """Base skorch warning."""


class DeviceWarning(SkorchWarning):
    """A problem with a device (e.g. CUDA) was detected."""
