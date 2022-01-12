'''Utility functions to convert nested dicts or lists to nested namespaces

Copy and pasted from https://stackoverflow.com/a/50491016
'''

from functools import singledispatch
from types import SimpleNamespace

@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]
