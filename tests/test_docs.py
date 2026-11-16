"""Every public name of ddma has a docstring (the API reference is generated from them)."""

import inspect

import pytest

import ddma
import ddma.invariance
import ddma.plotting

MODULES = [ddma, ddma.invariance, ddma.plotting]
try:
    import ddma.rl

    MODULES.append(ddma.rl)
except ImportError:  # gymnasium not installed
    pass


def _public():
    for mod in MODULES:
        for name in mod.__all__:
            obj = getattr(mod, name)
            yield f"{mod.__name__}.{name}", obj
            if inspect.isclass(obj):
                for attr, val in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(val, property):
                        val = val.fget
                    elif isinstance(val, (classmethod, staticmethod)):
                        val = val.__func__
                    if inspect.isfunction(val):
                        yield f"{mod.__name__}.{name}.{attr}", val


PUBLIC = list(_public())


@pytest.mark.parametrize("name, obj", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_has_docstring(name, obj):
    assert (obj.__doc__ or "").strip(), f"{name} has no docstring"
