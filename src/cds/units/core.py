"""Small dependency-free dimensional-analysis primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """Exponents of the seven SI base dimensions.

    Order: mass, length, time, electric current, temperature,
    amount of substance, luminous intensity.
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminous_intensity: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.current + other.current,
            self.temperature + other.temperature,
            self.amount + other.amount,
            self.luminous_intensity + other.luminous_intensity,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.current - other.current,
            self.temperature - other.temperature,
            self.amount - other.amount,
            self.luminous_intensity - other.luminous_intensity,
        )

    def __pow__(self, exponent: int) -> Dimension:
        return Dimension(
            self.mass * exponent,
            self.length * exponent,
            self.time * exponent,
            self.current * exponent,
            self.temperature * exponent,
            self.amount * exponent,
            self.luminous_intensity * exponent,
        )

    @property
    def is_dimensionless(self) -> bool:
        return self == Dimension()


@dataclass(frozen=True)
class Unit:
    """A linear unit represented by an SI scale and a dimension."""

    symbol: str
    scale: float
    dimension: Dimension

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("unit symbol must not be empty")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("unit scale must be finite and positive")

    def __mul__(self, other: Unit) -> Unit:
        return Unit(
            f"{self.symbol}*{other.symbol}",
            self.scale * other.scale,
            self.dimension * other.dimension,
        )

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(
            f"{self.symbol}/{other.symbol}",
            self.scale / other.scale,
            self.dimension / other.dimension,
        )

    def __pow__(self, exponent: int) -> Unit:
        return Unit(
            f"{self.symbol}^{exponent}",
            self.scale**exponent,
            self.dimension**exponent,
        )


@dataclass(frozen=True)
class Quantity:
    """A scalar value with a physical unit."""

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("quantity value must be finite")

    @property
    def si_value(self) -> float:
        return self.value * self.unit.scale

    def to(self, target: Unit) -> Quantity:
        """Convert to another unit with the same physical dimension."""
        if self.unit.dimension != target.dimension:
            raise ValueError("cannot convert between incompatible dimensions")
        return Quantity(self.si_value / target.scale, target)

    def __add__(self, other: Quantity) -> Quantity:
        converted = other.to(self.unit)
        return Quantity(self.value + converted.value, self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        converted = other.to(self.unit)
        return Quantity(self.value - converted.value, self.unit)

    def __mul__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        return Quantity(self.value * float(other), self.unit)

    def __rmul__(self, other: float) -> Quantity:
        return self * other

    def __truediv__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.unit / other.unit)
        return Quantity(self.value / float(other), self.unit)

    def __pow__(self, exponent: int) -> Quantity:
        return Quantity(self.value**exponent, self.unit**exponent)


def dimensions_compatible(*items: Dimension | Unit | Quantity) -> bool:
    """Return whether every item carries the same physical dimension."""
    if not items:
        raise ValueError("at least one item is required")

    def dimension(item: Dimension | Unit | Quantity) -> Dimension:
        if isinstance(item, Dimension):
            return item
        if isinstance(item, Unit):
            return item.dimension
        return item.unit.dimension

    first = dimension(items[0])
    return all(dimension(item) == first for item in items[1:])


DIMENSIONLESS = Unit("1", 1.0, Dimension())
KILOGRAM = Unit("kg", 1.0, Dimension(mass=1))
METER = Unit("m", 1.0, Dimension(length=1))
SECOND = Unit("s", 1.0, Dimension(time=1))
AMPERE = Unit("A", 1.0, Dimension(current=1))
KELVIN = Unit("K", 1.0, Dimension(temperature=1))
MOLE = Unit("mol", 1.0, Dimension(amount=1))
CANDELA = Unit("cd", 1.0, Dimension(luminous_intensity=1))

HERTZ = Unit("Hz", 1.0, SECOND.dimension**-1)
NEWTON = Unit("N", 1.0, KILOGRAM.dimension * METER.dimension / (SECOND.dimension**2))
PASCAL = Unit("Pa", 1.0, NEWTON.dimension / (METER.dimension**2))
JOULE = Unit("J", 1.0, NEWTON.dimension * METER.dimension)
WATT = Unit("W", 1.0, JOULE.dimension / SECOND.dimension)
COULOMB = Unit("C", 1.0, AMPERE.dimension * SECOND.dimension)
VOLT = Unit("V", 1.0, WATT.dimension / AMPERE.dimension)
