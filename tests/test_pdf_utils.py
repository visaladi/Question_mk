import pytest
from core import pdf_utils

def test_pdf_utils_module_exists():
    assert hasattr(pdf_utils, "__file__")
