"""Loanpy — linguistic toolkit for loanword detection and sound change."""

from loanpy.adapt import Adapt
from loanpy.cluster import Cluster
from loanpy.correspondences import get_correspondences, get_sound_correspondences
from loanpy.uralign import Uralign

cluster_cv = Cluster.cv  # backward compatibility
__all__ = [
    "Adapt",
    "Cluster",
    "Uralign",
    "cluster_cv",
    "get_correspondences",
    "get_sound_correspondences",
]
__version__ = "4.0.0"
