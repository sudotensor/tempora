from abc import ABC, abstractmethod


class Recorder(ABC):
    """Abstract base class for simulation recorders."""

    # Called at most once on every batch
    @abstractmethod
    def record(self, **kwargs):
        pass

    # Generate and return a dictionary of logged indicators
    @abstractmethod
    def emit(self):
        pass

    # Print the logged information to stdout
    @abstractmethod
    def print(self):
        pass