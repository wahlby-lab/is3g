from .is3g import is3g as is3g

try:
    from is3g._version import version as __version__
except ImportError:
    __version__ = "not-installed"

__ALL__ = ["is3g"]
