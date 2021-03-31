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

import pytest
import networkx as nx
import pandas as pd
import cudf

import cugraph
from cugraph.tests import utils

from .datasets import datasets


cugraph_graph_types = [cugraph.Graph, cugraph.DiGraph]
nx_cugraph_type_map = {cugraph.Graph : nx.Graph,
                       cugraph.DiGraph : nx.DiGraph,
                      }

# Create a list of pytest params using the datasets, using the pytest_marks attr
# of the dataset obj to create a pytest.mark list for use in specifying subsets
# of datasets with -m
datasets_as_params_list = [pytest.param(ds, marks=[getattr(pytest.mark, m)
                                                   for m in ds.pytest_marks])
                           for ds in datasets]


def call_csv_reader(csv_reader, dataset_obj):
    """
    Returns a dataframe by calling a compatible csv_reader on the csv specified
    by dataset_obj.
    """
    dtypes = {"0": dataset_obj.vertex_type,
              "1": dataset_obj.vertex_type,
             }
    names = ["0", "1"]
    if dataset_obj.weight_type is not None:
        dtypes["weight"] = dataset_obj.weight_type
        names.append("weight")

    print(f"\nREADING CSV {dataset_obj.path} USING {csv_reader}")
    df = csv_reader(dataset_obj.path,
                    delimiter=" ",
                    dtype=dtypes,
                    names=names,
                    header=None,
                   )
    return df


################################################################################
## Fixtures

@pytest.fixture(scope="session",
                params=datasets_as_params_list,
                ids=[f"dataset={ds.rel_path}" for ds in datasets])
def cudf_dataframe_from_dataset(request):
    """
    A tuple of (Dataset obj, cudf DataFrame) for each dataset read in.
    """
    dataset_obj = request.param

    df = call_csv_reader(cudf.read_csv, dataset_obj)
    return (dataset_obj, df)


@pytest.fixture(scope="session")
def cudf_pandas_dataframes_from_dataset(cudf_dataframe_from_dataset):
    """
    A tuple of (Dataset obj, cudf DataFrame, pandas DataFrame) for each
    cudf_dataframe_from_dataset.
    """
    (dataset_obj, df) = cudf_dataframe_from_dataset

    pdf = call_csv_reader(pd.read_csv, dataset_obj)
    return (dataset_obj, df, pdf)


@pytest.fixture(scope="session",
                params=cugraph_graph_types,
                ids=[f"type={gt.__name__}" for gt in cugraph_graph_types])
def cugraph_obj_from_dataset(cudf_dataframe_from_dataset, request):
    """
    A tuple of (Dataset obj, cudf DataFrame, cugraph obj) for each
    cugraph_graph_types, for each cudf_dataframe_from_dataset.
    """
    (dataset_obj, df) = cudf_dataframe_from_dataset
    graph_type = request.param

    G = graph_type()

    if dataset_obj.weight_type is not None:
        G.from_cudf_edgelist(df, source="0", destination="1", edge_attr="weight")
    else:
        G.from_cudf_edgelist(df, source="0", destination="1")

    return (dataset_obj, df, G)


@pytest.fixture(scope="session",
                params=cugraph_graph_types,
                ids=[f"type={gt.__name__}" for gt in cugraph_graph_types])
def cugraph_nx_objs_from_dataset(cudf_pandas_dataframes_from_dataset, request):
    """
    A tuple of (Dataset obj, DataFrame, cugraph obj, NetworkX obj) for each
    cugraph_graph_types, for each cudf_pandas_dataframes_from_dataset.

    The NetworkX obj is chosen based on the cugraph obj, as defined by
    nx_cugraph_type_map.
    """
    (dataset_obj, df, pdf) = cudf_pandas_dataframes_from_dataset
    graph_type = request.param

    G = graph_type()
    Gnx_type = nx_cugraph_type_map[graph_type]

    if dataset_obj.weight_type is not None:
        G.from_cudf_edgelist(df, source="0", destination="1",
                             edge_attr="weight")
        Gnx = nx.from_pandas_edgelist(pdf, source="0", target="1",
                                      edge_attr="weight",
                                      create_using=Gnx_type)
    else:
        G.from_cudf_edgelist(df, source="0", destination="1")
        Gnx = nx.from_pandas_edgelist(pdf, source="0", target="1",
                                      create_using=Gnx_type)

    return (dataset_obj, df, G, Gnx)
