"""Utility functions to convert to/from nested dicts or lists, and nested namespaces

References
----------
https://stackoverflow.com/a/50491016
"""

from functools import singledispatch
from types import SimpleNamespace


@singledispatch
def wrap_namespace(ob):
    return ob


@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{key: wrap_namespace(val) for key, val in ob.items()})


@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(val) for val in ob]


def unwrap_to_dict(sns: SimpleNamespace) -> dict:
    sns_as_dict = {}
    for key, val in sns.__dict__.items():
        if isinstance(val, SimpleNamespace):
            sns_as_dict[key] = unwrap_to_dict(val)
        else:
            sns_as_dict[key] = val

    return sns_as_dict
