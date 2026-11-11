"""Typed errors shared by data and configuration code."""


class ConfigError(ValueError):
    """Raised when a configuration is malformed or invalid."""


class DataIntegrityError(RuntimeError):
    """Raised when source data is missing, malformed, or inconsistent."""


class InsufficientDataError(ValueError):
    """Raised when available observations cannot support a requested operation."""


class LeakageError(RuntimeError):
    """Raised when patient groups overlap across partitions."""
