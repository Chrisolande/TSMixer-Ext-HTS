import json
import os

import numpy as np
import scipy.sparse as sp


def save_preprocess_bundle(bundle_dir: str, bundle: dict):
    """Save preprocessing bundle to bundle_dir containing bundle.npz, category_maps.json, price_stats.json."""
    os.makedirs(bundle_dir, exist_ok=True)

    npz_dict = {}
    if "calendar_scaler_mean" in bundle:
        npz_dict["calendar_scaler_mean"] = np.asarray(bundle["calendar_scaler_mean"], dtype=np.float64)
    if "calendar_scaler_scale" in bundle:
        npz_dict["calendar_scaler_scale"] = np.asarray(bundle["calendar_scaler_scale"], dtype=np.float64)
    if "weights" in bundle:
        npz_dict["weights"] = np.asarray(bundle["weights"], dtype=np.float64)
    if "scaling_factors" in bundle:
        npz_dict["scaling_factors"] = np.asarray(bundle["scaling_factors"], dtype=np.float64)

    if "S_matrix" in bundle:
        s_mat = bundle["S_matrix"]
        if sp.issparse(s_mat):
            s_csr = s_mat.tocsr()
            npz_dict["s_indices"] = s_csr.indices.astype(np.int32)
            npz_dict["s_indptr"] = s_csr.indptr.astype(np.int32)
            npz_dict["s_data"] = s_csr.data.astype(np.float32)
            npz_dict["s_shape"] = np.array(s_csr.shape, dtype=np.int32)
    elif "s_indices" in bundle:
        npz_dict["s_indices"] = np.asarray(bundle["s_indices"], dtype=np.int32)
        npz_dict["s_indptr"] = np.asarray(bundle["s_indptr"], dtype=np.int32)
        npz_dict["s_data"] = np.asarray(bundle.get("s_data", np.ones_like(bundle["s_indices"])), dtype=np.float32)
        npz_dict["s_shape"] = np.asarray(bundle["s_shape"], dtype=np.int32)

    for k, v in bundle.items():
        if k not in (
            "calendar_scaler_mean",
            "calendar_scaler_scale",
            "weights",
            "scaling_factors",
            "S_matrix",
            "s_indices",
            "s_indptr",
            "s_data",
            "s_shape",
            "category_maps",
            "price_stats",
        ):
            if isinstance(v, (np.ndarray, list)):
                npz_dict[k] = np.asarray(v)

    np.savez_compressed(os.path.join(bundle_dir, "bundle.npz"), **npz_dict)

    cat_maps = bundle.get("category_maps", {})
    with open(os.path.join(bundle_dir, "category_maps.json"), "w") as f:
        json.dump(cat_maps, f, indent=2)

    price_stats = bundle.get("price_stats", {})
    with open(os.path.join(bundle_dir, "price_stats.json"), "w") as f:
        json.dump(price_stats, f, indent=2)


def load_preprocess_bundle(bundle_dir: str) -> dict:
    """Load preprocessing bundle from bundle_dir."""
    npz_path = os.path.join(bundle_dir, "bundle.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Preprocessing bundle archive not found at {npz_path}")

    npz = np.load(npz_path)
    bundle = {k: npz[k] for k in npz.files}

    if "s_indices" in bundle and "s_indptr" in bundle and "s_shape" in bundle:
        s_data = bundle.get("s_data", np.ones_like(bundle["s_indices"], dtype=np.float32))
        s_shape = tuple(bundle["s_shape"])
        bundle["S_matrix"] = sp.csr_matrix((s_data, bundle["s_indices"], bundle["s_indptr"]), shape=s_shape)

    cat_path = os.path.join(bundle_dir, "category_maps.json")
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            bundle["category_maps"] = json.load(f)
    else:
        bundle["category_maps"] = {}

    price_path = os.path.join(bundle_dir, "price_stats.json")
    if os.path.exists(price_path):
        with open(price_path) as f:
            bundle["price_stats"] = json.load(f)
    return bundle
