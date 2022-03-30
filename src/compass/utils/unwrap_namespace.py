'''Utility functions to convert nested namespaces to nested dict
'''

from types import SimpleNamespace

def unwrap_to_dict(sns: SimpleNamespace) -> dict:
    sns_as_dict = {}
    for k, v in sns.__dict__.items():
        if isinstance(v, SimpleNamespace):
            unwrap_to_dict(v)
        else:
            sns_dict[k] = v

    return sns_as_dict
