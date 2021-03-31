# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import pytest

# path is relative to RAPIDS_DATASET_ROOT_DIR if specified, else the default
# datasets dir
# weighted=False and weight_type=float32 means a weight column is present in the
# data, set to 1.0

# FIXME: add a float64 weight_type

dataset_metadata = [
["path",                     "vertices", "edges",   "directed", "weighted", "self_loops", "isolated_vertices", "multi_edges", "vertex_type", "weight_type", "pytest_marks",],

["karate.csv",               34,         156,       False,      False,      False,        False,               False,         "int32",       "float32",     "small"        ],
["dolphins.csv",             62,         318,       False,      False,      False,        False,               False,         "int32",       "float32",     "small"        ],
["netscience.csv",           1589,       5484,      False,      True,       False,        False,               False,         "int32",       "float32",     None           ],
["email-Eu-core.csv",        1005,       25571,     True,       False,      False,        False,               False,         "int32",       "float32",     None           ],
["karate_multi_edge.csv",    34,         160,       False,      False,      False,        False,               True,          "int32",       "float32",     None           ],
["dolphins_multi_edge.csv",  62,         325,       False,      False,      False,        False,               True,          "int32",       "float32",     None           ],
["karate_s_loop.csv",        34,         160,       False,      False,      True,         False,               False,         "int32",       "float32",     None           ],
["dolphins_s_loop.csv",      62,         321,       False,      False,      True,         False,               False,         "int32",       "float32",     None           ],
#["karate_mod.mtx",           37,         156,       False,      False,      False,        True,                False,         "int32",       None,          None           ],
#["karate_str.mtx",           34,         156,       False,      True,       False,        False,               False,         "string",      "int32",       None           ],
]

# This assumes this file resides in a specific place in the source dir
# hierarchy under the cugraph root. If ever moved, this must be updated!
__default_dataset_root_dir = \
    Path(Path(__file__).resolve().parent / "../../../../datasets").resolve()
rapids_dataset_root_dir = os.getenv("RAPIDS_DATASET_ROOT_DIR",
                                    __default_dataset_root_dir)


class Dataset(dict):
    """
    Dict object aware of dataset-specific metadata.
    """
    valid_keys = set(dataset_metadata[0])

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        # Enforce specific keys to eliminate typos when specifying metadata
        invalid_keys = set(self.keys()) - self.valid_keys
        if invalid_keys:
            raise KeyError(invalid_keys.pop()) # any invalid key will do

        # Ensure path is always an abs path to allow access from any CWD
        # Save rel_path to use for reporting the name to console
        if "path" in self.keys():
            p = Path(self["path"])
            self.rel_path = p
            if not p.is_absolute():
                self["path"] = rapids_dataset_root_dir / p
        else:
            self.rel_path = None

        # pytest_marks must be a list to be properly used in pytest.param()
        pytest_marks = self.get("pytest_marks") or []
        if pytest_marks:
            pytest_marks = pytest_marks.split(",")
        self["pytest_marks"] = pytest_marks

    def __getattr__(self, attr):
        """
        Allows for attribute access from the dictionary
        """
        if attr in self.valid_keys:
            return self.get(attr)
        raise AttributeError


__metadata_keys = dataset_metadata[0]
__metadata = dataset_metadata[1:]
datasets = [Dataset(zip(__metadata_keys, vals)) for vals in __metadata]
