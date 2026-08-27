import os
import tempfile

import pytest
import torch

from hier_forecast.api.schemas.request import SeriesKey
from hier_forecast.api.store import InferenceStore
from hier_forecast.data_processing.bundle import save_preprocess_bundle
from hier_forecast.data_processing.pipeline import preprocess_m5


def test_api_store_dynamic_exogenous_assembly():
    sample_dir = "data/m5_sample"
    if not os.path.exists(sample_dir):
        pytest.skip(f"Sample data directory {sample_dir} not found.")

    with tempfile.TemporaryDirectory() as bundle_dir:
        data_dict = preprocess_m5(sample_dir, train_days=1886)
        save_preprocess_bundle(bundle_dir, data_dict)

        store = InferenceStore(
            category_maps=data_dict["category_maps"],
            snapshot_dir=sample_dir,
            bundle_dir=bundle_dir,
        )

        first_item = list(data_dict["category_maps"]["item_id"].keys())[0]
        first_store = list(data_dict["category_maps"]["store_id"].keys())[0]
        item = SeriesKey(
            store_id=first_store,
            item_id=first_item,
            past_sales=[1.0] * 35,
        )

        tensors = store.build_tensors(item, as_of_date="2016-04-25")

        # Verify not all zeros (actual real calendar + price features injected)
        assert tensors["x_hist"].abs().sum().item() > 0.0
        assert tensors["z_futr"].abs().sum().item() > 0.0
        assert not torch.all(tensors["x_hist"] == 0.0)
        assert not torch.all(tensors["z_futr"] == 0.0)

        # Check shapes
        assert tensors["x_hist"].shape == (35, 10)
        assert tensors["z_futr"].shape == (28, 10)
        assert tensors["s_cat"].shape == (5,)
        assert tensors["s_cont"].shape == (1,)


def test_api_store_unknown_store_or_item_raises():
    store = InferenceStore(
        category_maps={"store_id": {"CA_1": 0}, "item_id": {"HOBBIES_1_001": 0}},
        snapshot_dir="data/m5_sample",
    )

    invalid_store = SeriesKey(store_id="INVALID_STORE", item_id="HOBBIES_1_001")
    with pytest.raises(KeyError, match="Unknown store_id"):
        store.build_tensors(invalid_store)

    invalid_item = SeriesKey(store_id="CA_1", item_id="UNKNOWN_ITEM")
    with pytest.raises(KeyError, match="Unknown item_id"):
        store.build_tensors(invalid_item)
