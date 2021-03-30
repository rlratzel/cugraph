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
import cudf

import cugraph
from cugraph.tests import utils

from .datasets import datasets


cugraph_graph_types = [cugraph.Graph, cugraph.DiGraph]
nx_cugraph_type_map = {cugraph.Graph : nx.Graph,
                       cugraph.DiGraph : nx.DiGraph,
                      }

def read_csv_dataset_using(csv_reader, dataset_obj):
    """
    Returns a dataframe by calling csv_reader on the csv specified by
    dataset_obj.
    """
    dtypes = {"0": dataset_obj.vertex_type,
              "1": dataset_obj.vertex_type,
             }
    names = ["0", "1"]
    if dataset_obj.weight_type is not None:
        dtypes["weight"] = dataset_obj.weight_type
        names.append("weight")

    print(f"\nREADING CSV {dataset_obj.path} USING {csv_reader}\n")
    df = csv_reader(dataset_obj.path,
                    delimiter=" ",
                    dtype=dtypes,
                    names=names,
                    header=None,
                   )
    return df


################################################################################
## Fixtures

@pytest.fixture(scope="package",
                params=datasets,
                ids=[f"dataset={ds.rel_path}" for ds in datasets])
def cudf_dataframe_from_dataset(request):
    dataset_obj = request.param

    df = read_csv_dataset_using(cudf.read_csv, dataset_obj)
    return (dataset_obj, df)


@pytest.fixture(scope="package",
                params=cugraph_graph_types,
                ids=[f"type={gt.__name__}" for gt in cugraph_graph_types])
def cugraph_obj_from_dataset(cudf_dataframe_from_dataset, request):
    (dataset_obj, df) = cudf_dataframe_from_dataset
    graph_type = request.param

    G = graph_type()

    if dataset_obj.weight_type is not None:
        G.from_cudf_edgelist(df, source="0", destination="1", edge_attr="weight")
    else:
        G.from_cudf_edgelist(df, source="0", destination="1")

    return (dataset_obj, df, G)


@pytest.fixture(scope="package",
                params=cugraph_graph_types,
                ids=[f"type={gt.__name__}" for gt in cugraph_graph_types])
def cugraph_nx_obj_from_dataset(cudf_dataframe_from_dataset, request):
    (dataset_obj, df) = cudf_dataframe_from_dataset
    graph_type = request.param

    G = graph_type()
    Gnx = nx_cugraph_type_map[graph_type]()
    pdf = read_csv_dataset_using(pd.read_csv, dataset_obj)

    if dataset_obj.weight_type is not None:
        G.from_cudf_edgelist(df, source="0", destination="1",
                             edge_attr="weight")
        Gnx.from_pandas_edgelist(pdf, source="0", destination="1",
                                 edge_attr="weight")
    else:
        G.from_cudf_edgelist(df, source="0", destination="1")
        Gnx.from_pandas_edgelist(pdf, source="0", destination="1")

    return (dataset_obj, df, G, Gnx)
