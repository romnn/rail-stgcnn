import pytest

from cargonet.constants.numbers import MININT


def test_minint_consistency():
    assert -9223372036854775808 == MININT
