"""Loanpy — linguistic toolkit for loanword detection and sound change."""

from loanpy.core import Cluster, Uralign

cluster_cv = Cluster.cv  # backward compatibility
__all__ = ["Cluster", "cluster_cv", "Uralign"]
__version__ = "4.0.0"
