from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error code and description."""

    code: str = Field(..., json_schema_extra={"example": "CATEGORY_NOT_FOUND"})
    message: str = Field(..., json_schema_extra={"example": "Store or item ID not recognized in categorical mapping"})


class Quantiles(BaseModel):
    """Predicted sales quantile series (28 daily values each)."""

    p10: list[int | float] = Field(..., description="10th percentile forecast quantile")
    p50: list[int | float] = Field(..., description="50th percentile forecast quantile (median)")
    p90: list[int | float] = Field(..., description="90th percentile forecast quantile")
    median: list[int | float] | None = Field(default=None, description="Explicit alias for p50 median")


class ItemForecastResult(BaseModel):
    """Individual time series forecast result or item error."""

    store_id: str
    item_id: str
    status: str = Field(
        ..., json_schema_extra={"example": "success"}, description="Result status ('success' or 'error')"
    )
    mean: list[float] | None = Field(default=None, description="Predicted 28-day mean sales (mu)")
    median: list[int | float] | None = Field(default=None, description="Predicted 28-day exact discrete median")
    dispersion: list[float] | None = Field(default=None, description="Predicted Negative Binomial alpha parameter")
    quantiles: Quantiles | None = Field(default=None, description="Quantile distribution bounds")
    error_detail: ErrorDetail | None = Field(default=None, description="Populated when status is 'error'")


class ForecastResponse(BaseModel):
    """Batch forecast response payload."""

    as_of_date: str
    horizon_days: int = Field(default=28)
    results: list[ItemForecastResult]


class ErrorResponse(BaseModel):
    """Structured top-level error response payload."""

    code: str = Field(..., json_schema_extra={"example": "VALIDATION_ERROR"})
    message: str = Field(..., json_schema_extra={"example": "Invalid request payload"})
    details: dict | None = Field(default=None)
