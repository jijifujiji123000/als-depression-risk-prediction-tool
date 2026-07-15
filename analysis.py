"""Compatibility definitions required when loading the locked model artifacts.

The serialized artifacts contain a fitted FastDevelopmentOnlyMI instance under
the historical module name ``analysis``. The web tool accepts complete required
inputs, so the fitted imputer is retained for provenance but is never refitted
or invoked during prediction.
"""


class FastDevelopmentOnlyMI:
    """Compatibility shell for the fitted development-only imputer."""

    pass
