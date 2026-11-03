"""Calibration of ngsrng's automatic choices on this machine (``python -m ngsrng.calibrate``).

``cases`` defines the candidates and profiles, ``worker`` measures one group of them in an isolated process,
``runner`` orchestrates the groups (time budgets, rejection of bad candidates, resume), ``recommend`` turns the records
into the recommendations ``ngsrng.tuning`` installs, and ``report`` writes the human-readable report (in full with
pandas and matplotlib, in short without). The ``research`` profile also runs the suites that inform the library's
design rather than its tuning (posteriors, distributions, low rank, covariance diagnostics, continuum accuracy).
"""
