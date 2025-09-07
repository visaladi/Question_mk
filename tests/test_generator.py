import pytest
from core import generator

def test_generator_module_exists():
    assert hasattr(generator, "__file__")
