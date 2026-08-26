from __future__ import annotations

from .factor_base import LinearFactorExpectedReturnModel


class FamaFrench3ExpectedReturn(LinearFactorExpectedReturnModel):
    factor_columns = ("MKT_RF", "SMB", "HML")