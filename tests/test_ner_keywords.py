import pytest
from extras import ner_keywords

def test_ner_keywords_module_exists():
    assert hasattr(ner_keywords, "__file__")
