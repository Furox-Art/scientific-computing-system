"""Dimensional contracts connecting :mod:`cds.units` to model fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from cds.units import Quantity, Unit

NumericOrQuantity = float | int | Quantity


@dataclass
class ModelUnitContract:
    """Declared units for model inputs, fitted parameters, and target equations.

    Bare numeric values are interpreted in their declared unit by default so
    existing numerical datasets remain usable.  Set ``require_quantities=True``
    when every declared dimensional value must arrive as an explicit
    :class:`~cds.units.Quantity`.
    """

    variable_units: dict[str, Unit] = field(default_factory=dict)
    parameter_units: dict[str, Unit] = field(default_factory=dict)
    target_units: dict[str, Unit] = field(default_factory=dict)
    require_quantities: bool = False

    def __post_init__(self) -> None:
        for mapping_name, mapping in (
            ("variable_units", self.variable_units),
            ("parameter_units", self.parameter_units),
            ("target_units", self.target_units),
        ):
            if any(not name.strip() for name in mapping):
                raise ValueError(f"{mapping_name} must not contain empty names")

    @staticmethod
    def _normalize(
        name: str,
        value: NumericOrQuantity,
        unit: Unit | None,
        *,
        require_quantity: bool,
    ) -> float:
        if isinstance(value, Quantity):
            if unit is None:
                raise ValueError(f"no unit declared for dimensional value {name!r}")
            normalized = value.to(unit).value
        else:
            if unit is not None and require_quantity:
                raise ValueError(f"value {name!r} must be supplied as Quantity")
            normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"value {name!r} must be finite")
        return normalized

    def normalize_environment(self, env: Mapping[str, NumericOrQuantity]) -> dict[str, float]:
        """Convert an observation environment into declared model units."""
        return {
            name: self._normalize(
                name,
                value,
                self.variable_units.get(name),
                require_quantity=self.require_quantities and name in self.variable_units,
            )
            for name, value in env.items()
        }

    def normalize_target(self, label: str, value: NumericOrQuantity) -> float:
        """Convert an observed target into the target equation's declared unit."""
        return self._normalize(
            label,
            value,
            self.target_units.get(label),
            require_quantity=self.require_quantities and label in self.target_units,
        )

    def normalize_parameter(self, name: str, value: NumericOrQuantity) -> float:
        """Convert a starting parameter value into its declared unit."""
        return self._normalize(
            name,
            value,
            self.parameter_units.get(name),
            require_quantity=self.require_quantities and name in self.parameter_units,
        )

    def validate_declarations(
        self,
        *,
        variable_names: set[str],
        parameter_names: set[str],
        target_labels: set[str],
    ) -> None:
        """Reject unit declarations that do not correspond to the model surface."""
        unknown_variables = set(self.variable_units) - variable_names
        unknown_parameters = set(self.parameter_units) - parameter_names
        unknown_targets = set(self.target_units) - target_labels
        if unknown_variables:
            raise ValueError(f"unit contract declares unknown variables: {sorted(unknown_variables)}")
        if unknown_parameters:
            raise ValueError(f"unit contract declares unknown parameters: {sorted(unknown_parameters)}")
        if unknown_targets:
            raise ValueError(f"unit contract declares unknown targets: {sorted(unknown_targets)}")
