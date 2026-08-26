from __future__ import annotations

from .factor_base import LinearFactorExpectedReturnModel


class CAPMExpectedReturn(LinearFactorExpectedReturnModel):
    factor_columns = ("MKT_RF",)