from __future__ import annotations

from .factor_base import LinearFactorExpectedReturnModel


class CarhartExpectedReturn(LinearFactorExpectedReturnModel):
    factor_columns = ("MKT_RF", "SMB", "HML", "MOM")