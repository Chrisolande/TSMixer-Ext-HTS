from pydantic import BaseModel, Field

_EXAMPLE_35 = [float(v) for v in [0,1,2,2,0,1,0,1,0,2,0,0,1,2,0,3,3,5,1,2,0,1,0,1,1,2,2,3,2,0,2,1,2,4,2]]


class SeriesKey(BaseModel):
    """Identifier and optional past sales for a single time series."""

    store_id: str = Field(..., json_schema_extra={"example": "CA_1"}, description="M5 store identifier")
    item_id: str = Field(..., json_schema_extra={"example": "HOBBIES_1_001"}, description="M5 item identifier")
    past_sales: list[float] | None = Field(
        default=None,
        description="Optional 35-day historical sales override (must be exactly 35 values). "
        "Omit or set null to use the on-disk snapshot automatically.",
    )


class ForecastRequest(BaseModel):
    """Batch forecast request payload."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Default - use on-disk snapshot (recommended)",
                    "value": {
                        "as_of_date": "2016-04-25",
                        "items": [{"store_id": "CA_1", "item_id": "HOBBIES_1_001"}],
                        "return_quantiles": True,
                    },
                },
                {
                    "summary": "Manual override - supply your own 35-day window",
                    "value": {
                        "as_of_date": "2016-04-25",
                        "items": [
                            {
                                "store_id": "CA_1",
                                "item_id": "HOBBIES_1_001",
                                "past_sales": _EXAMPLE_35,
                            }
                        ],
                        "return_quantiles": True,
                    },
                },
            ]
        }
    }

    as_of_date: str = Field(
        ..., json_schema_extra={"example": "2016-04-25"}, description="Forecast origin date (YYYY-MM-DD)"
    )
    items: list[SeriesKey] = Field(..., min_length=1, max_length=1000, description="List of series to forecast")
    return_quantiles: bool = Field(default=True, description="Whether to include p10, p50, p90 quantile bounds")
