from .isseg import isseg as isseg

try:
    from isseg._version import version as __version__
except ImportError:
    __version__ = "not-installed"

__ALL__ = ["isseg"]
