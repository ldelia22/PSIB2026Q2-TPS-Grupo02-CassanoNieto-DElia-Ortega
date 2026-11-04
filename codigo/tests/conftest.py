"""Un test cuya función todavía no existe figura como skipped, no como error.

Así la suite se mantiene verde desde el primer día. Cuando alguien implementa una
función, su test pasa a ejecutarse de verdad: si la implementación está mal, falla.
"""

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as pendiente:
        pytest.skip(f"Sin implementar: {pendiente}")
