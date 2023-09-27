# Copyright (c) 2023, NVIDIA CORPORATION.
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
import time

import networkx as nx
import pandas as pd
import pytest
from cugraph import datasets

# FIXME: promote these to cugraph.datasets
from cugraph_benchmarking.params import (
    hollywood,
    europe_osm,
    cit_patents,
    soc_livejournal,
)

################################################################################
# Fixtures and helpers
backend_params = ["cugraph", None]

small_dataset_params = [datasets.karate,
                        datasets.netscience,
                        datasets.email_Eu_core,
                        ]
medium_dataset_params = [cit_patents,
                         hollywood,
                         europe_osm,
                         ]
large_dataset_params = [soc_livejournal,
                        ]

def nx_graph_from_dataset(dataset_obj):
    """
    Read the dataset specified by the dataset_obj and create and return a
    nx.Graph or nx.DiGraph instance based on the dataset is_directed metadata.
    """
    create_using = nx.DiGraph if dataset_obj.metadata["is_directed"] else nx.Graph
    names = dataset_obj.metadata["col_names"]
    dtypes = dataset_obj.metadata["col_types"]
    if isinstance(dataset_obj.metadata["header"], int):
        header = dataset_obj.metadata["header"]
    else:
        header = None

    pandas_edgelist = pd.read_csv(dataset_obj.get_path(),
                                  delimiter=dataset_obj.metadata["delim"],
                                  names=names,
                                  dtype=dict(zip(names, dtypes)),
                                  header=header,
                                  )
    G = nx.from_pandas_edgelist(pandas_edgelist,
                                source=names[0],
                                target=names[1],
                                create_using=create_using)
    return G


@pytest.fixture(scope="module", params=small_dataset_params,
                ids=lambda ds: f"ds={str(ds)}")
def small_graph_obj(request):
    dataset = request.param
    return nx_graph_from_dataset(dataset)


@pytest.fixture(scope="module", params=medium_dataset_params,
                ids=lambda ds: f"ds={str(ds)}")
def medium_graph_obj(request):
    dataset = request.param
    return nx_graph_from_dataset(dataset)


@pytest.fixture(scope="module", params=large_dataset_params,
                ids=lambda ds: f"ds={str(ds)}")
def large_graph_obj(request):
    dataset = request.param
    return nx_graph_from_dataset(dataset)


# FIXME: this is needed for networkx <3.2, networkx >=3.2 simply allows the
# backend to be specified using a parameter. For now, use the same technique
# for all NX versions
try:
    from networkx.classes import backends  # NX <3.2
    _using_legacy_dispatcher = True
except ImportError:
    backends = None
    _using_legacy_dispatcher = False


def get_legacy_backend_selector(backend_name):
    """
    Returns a callable that wraps an algo function with either the default
    dispatch decorator, or the "testing" decorator which unconditionally
    dispatches.
    This is only supported for NetworkX <3.2
    """
    backends.plugin_name = "cugraph"
    orig_dispatch = backends._dispatch
    testing_dispatch = backends.test_override_dispatch

    # Testing with the networkx <3.2 dispatch mechanism is based on decorating
    # networkx APIs. The decorator is either one that only uses a backend if
    # the input graph type is for that backend (the default decorator), or the
    # "testing" decorator, which unconditionally converts a graph type to the
    # type needed by the backend then calls the backend. If the cugraph backend
    # is specified, create a callable that decorates the benchmarked function
    # with the testing decorator.
    #
    # Because both the default and testing decorators assume they are only
    # applied once and do bookkeeping to ensure algos are not registered
    # multiple times, the callable also clears bookkeeping so the decorators
    # can be reapplied multiple times. This is obviously a hack and networkx
    # >=3.2 makes this use case properly supported.
    if backend_name == "cugraph":
        def wrapper(*args, **kwargs):
            backends._registered_algorithms = {}
            return testing_dispatch(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            backends._registered_algorithms = {}
            return orig_dispatch(*args, **kwargs)

    return wrapper


def get_backend_selector(backend_name):
    """
    Returns a callable that wraps an algo function in order to set the
    "backend" kwarg on it.
    This is only supported for NetworkX >= 3.2
    """
    def get_callable_for_func(func):
        def wrapper(*args, **kwargs):
            kwargs["backend"] = backend_name
            return func(*args, **kwargs)
        return wrapper

    return get_callable_for_func


@pytest.fixture(scope="module", params=backend_params,
                ids=lambda backend: f"backend={backend}")
def backend_selector(request):
    """
    Returns a callable that takes a function algo and wraps it in another
    function that calls the algo using the appropriate backend.
    """
    backend_name = request.param
    if _using_legacy_dispatcher:
        return get_legacy_backend_selector(backend_name)
    else:
        return get_backend_selector(backend_name)


################################################################################
# Benchmarks
normalized_params = [True, False]
k_params = [10, 100, 1000]


@pytest.mark.parametrize("normalized", normalized_params, ids=lambda norm: f"{norm=}")
def bench_betweenness_centrality_small(benchmark,
                                       small_graph_obj,
                                       backend_selector,
                                       normalized):
    result = benchmark(backend_selector(nx.betweenness_centrality),
                       small_graph_obj, weight=None, normalized=normalized)
    assert type(result) is dict


@pytest.mark.parametrize("normalized", normalized_params, ids=lambda norm: f"{norm=}")
def bench_edge_betweenness_centrality_small(benchmark,
                                            small_graph_obj,
                                            backend_selector,
                                            normalized):
    result = benchmark(backend_selector(nx.edge_betweenness_centrality),
                       small_graph_obj, weight=None, normalized=normalized)
    assert type(result) is dict


@pytest.mark.parametrize("k", k_params, ids=lambda k: f"{k=}")
def bench_betweenness_centrality_medium(benchmark,
                                        medium_graph_obj,
                                        backend_selector,
                                        k):
    result = benchmark(backend_selector(nx.betweenness_centrality),
                       medium_graph_obj, weight=None, k=k)
    assert type(result) is dict


@pytest.mark.parametrize("k", k_params, ids=lambda k: f"{k=}")
def bench_edge_betweenness_centrality_medium(benchmark,
                                             medium_graph_obj,
                                             backend_selector,
                                             k):
    result = benchmark(backend_selector(nx.edge_betweenness_centrality),
                       medium_graph_obj, weight=None, k=k)
    assert type(result) is dict


def bench_louvain_communities_small(benchmark,
                                    small_graph_obj,
                                    backend_selector):
    # The cugraph backend for louvain_communities only supports undirected graphs
    if isinstance(small_graph_obj, nx.DiGraph):
        G = small_graph_obj.to_undirected()
    else:
        G = small_graph_obj
    result = benchmark(backend_selector(nx.community.louvain_communities), G)
    assert type(result) is list


def bench_louvain_communities_medium(benchmark,
                                     medium_graph_obj,
                                     backend_selector):
    # The cugraph backend for louvain_communities only supports undirected graphs
    if isinstance(medium_graph_obj, nx.DiGraph):
        G = medium_graph_obj.to_undirected()
    else:
        G = medium_graph_obj
    result = benchmark(backend_selector(nx.community.louvain_communities), G)
    assert type(result) is list
