"""Component of a node (Learner, Aggregator, Communication Protocol...)."""

from abc import ABCMeta
from collections.abc import Callable
from typing import Any


def allow_no_addr_check(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate to mark a method as exempt from the addr check."""
    method.__no_addr_check__ = True  # type: ignore
    return method


class AddrRequiredMeta(ABCMeta):
    """Metaclass to ensure that the addr is set before any method is called, unless the method is marked with @allow_no_addr_check."""

    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> Any:
        """Create a new class with methods wrapped to ensure the addr is set."""
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and attr_name != "set_addr" and attr_name != "__init__":
                dct[attr_name] = cls.ensure_addr_set(attr_value)
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def ensure_addr_set(method: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a method to ensure the addr is set before it is called, unless the method is decorated with @allow_no_addr_check."""

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if hasattr(method, "__no_addr_check__"):
                # Method is marked as exempt, allow execution without addr check
                return method(self, *args, **kwargs)
            if not hasattr(self, "addr") or self.addr == "":
                raise ValueError("Address must be set before calling this method.")
            return method(self, *args, **kwargs)

        return wrapper

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create an instance of the class and initialize the addr attribute to an empty string."""
        instance = super().__call__(*args, **kwargs)
        instance.addr = ""
        return instance

    def set_addr(cls, instance: Any, addr: str) -> None:
        """Set the addr of the instance."""
        instance.addr = addr


class NodeComponent(metaclass=AddrRequiredMeta):
    """
    Component of a node (Learner, Aggregator, Communication Protocol...).

    Attributes:
        addr: The address of the node (must be a non-empty string).

    """

    addr: str

    def set_addr(self, addr: str) -> str:
        """Set the addr of the node."""
        AddrRequiredMeta.set_addr(NodeComponent, self, addr)
        return self.addr
