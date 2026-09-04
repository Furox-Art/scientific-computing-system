"""Tests for dependency-free SI units and dimensional analysis."""

from __future__ import annotations

import math

import pytest

from cds.units import (
    AMPERE,
    COULOMB,
    DIMENSIONLESS,
    HERTZ,
    JOULE,
    KILOGRAM,
    METER,
    NEWTON,
    PASCAL,
    SECOND,
    VOLT,
    WATT,
    Dimension,
    Quantity,
    Unit,
    dimensions_compatible,
)


def test_dimension_algebra_and_dimensionless() -> None:
    mass = Dimension(mass=1)
    length = Dimension(length=1)
    time = Dimension(time=1)

    force = mass * length / (time**2)
    assert force == NEWTON.dimension
    assert (length / time) * time == length
    assert (length**0).is_dimensionless
    assert not length.is_dimensionless


def test_unit_validation_and_algebra() -> None:
    centimeter = Unit("cm", 0.01, METER.dimension)
    speed = METER / SECOND
    area = METER**2
    combined = KILOGRAM * METER / (SECOND**2)

    assert centimeter.scale == 0.01
    assert speed.dimension == METER.dimension / SECOND.dimension
    assert area.dimension == METER.dimension**2
    assert combined.dimension == NEWTON.dimension

    with pytest.raises(ValueError, match="symbol"):
        Unit(" ", 1.0, Dimension())
    with pytest.raises(ValueError, match="scale"):
        Unit("bad", 0.0, Dimension())
    with pytest.raises(ValueError, match="scale"):
        Unit("bad", math.inf, Dimension())


def test_quantity_conversion_addition_and_subtraction() -> None:
    centimeter = Unit("cm", 0.01, METER.dimension)
    meter = Quantity(1.0, METER)
    fifty_cm = Quantity(50.0, centimeter)

    assert meter.si_value == 1.0
    assert meter.to(centimeter).value == pytest.approx(100.0)
    assert (meter + fifty_cm).value == pytest.approx(1.5)
    assert (meter - fifty_cm).value == pytest.approx(0.5)

    with pytest.raises(ValueError, match="incompatible"):
        meter.to(SECOND)
    with pytest.raises(ValueError, match="quantity value"):
        Quantity(math.nan, METER)


def test_quantity_multiplication_division_power_and_scalars() -> None:
    distance = Quantity(10.0, METER)
    duration = Quantity(2.0, SECOND)
    mass = Quantity(3.0, KILOGRAM)

    speed = distance / duration
    assert speed.value == 5.0
    assert speed.unit.dimension == METER.dimension / SECOND.dimension

    momentum = mass * speed
    assert momentum.value == 15.0
    assert momentum.unit.dimension == KILOGRAM.dimension * METER.dimension / SECOND.dimension

    doubled = distance * 2
    assert doubled == Quantity(20.0, METER)
    assert 3 * distance == Quantity(30.0, METER)
    assert distance / 2 == Quantity(5.0, METER)

    area = distance**2
    assert area.value == 100.0
    assert area.unit.dimension == METER.dimension**2


def test_predefined_derived_units_have_correct_dimensions() -> None:
    assert HERTZ.dimension == SECOND.dimension**-1
    assert NEWTON.dimension == KILOGRAM.dimension * METER.dimension / (SECOND.dimension**2)
    assert PASCAL.dimension == NEWTON.dimension / (METER.dimension**2)
    assert JOULE.dimension == NEWTON.dimension * METER.dimension
    assert WATT.dimension == JOULE.dimension / SECOND.dimension
    assert COULOMB.dimension == AMPERE.dimension * SECOND.dimension
    assert VOLT.dimension == WATT.dimension / AMPERE.dimension
    assert DIMENSIONLESS.dimension.is_dimensionless


def test_dimensions_compatible_accepts_dimensions_units_and_quantities() -> None:
    centimeter = Unit("cm", 0.01, METER.dimension)
    assert dimensions_compatible(METER.dimension, METER, Quantity(2.0, centimeter))
    assert not dimensions_compatible(METER, SECOND)
    assert dimensions_compatible(DIMENSIONLESS)

    with pytest.raises(ValueError, match="at least one"):
        dimensions_compatible()
