# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import random

import pytest
import networkx as nx
import networkx.algorithms.centrality.betweenness as nxacb
import cupy

import cugraph

_seed = 42  # seed for random()
_default_epsilon = 1e-6

################################################################################
## Test helpers

def _compare_bfs(cugraph_df, nx_distances, source):
    # This call should only contain 3 columns:
    # 'vertex', 'distance', 'predecessor'
    # It also confirms wether or not 'sp_counter' has been created by the call
    # 'sp_counter' triggers atomic operations in BFS, thus we want to make
    # sure that it was not the case
    # NOTE: 'predecessor' is always returned while the C++ function allows to
    # pass a nullptr
    assert len(cugraph_df.columns) == 3, (
        "The result of the BFS has an invalid " "number of columns"
    )
    cu_distances = {
        vertex: dist
        for vertex, dist in zip(
            cugraph_df["vertex"].to_array(), cugraph_df["distance"].to_array()
        )
    }
    cu_predecessors = {
        vertex: dist
        for vertex, dist in zip(
                cugraph_df["vertex"].to_array(),
                cugraph_df["predecessor"].to_array()
        )
    }

    # FIXME: The following only verifies vertices that were reached
    #       by cugraph's BFS.
    # We assume that the distances are given back as integers in BFS
    # max_val = np.iinfo(df['distance'].dtype).max
    # Unreached vertices have a distance of max_val

    missing_vertex_error = 0
    distance_mismatch_error = 0
    invalid_predecessor_error = 0
    for vertex in nx_distances:
        if vertex in cu_distances:
            result = cu_distances[vertex]
            expected = nx_distances[vertex]
            if result != expected:
                print(
                    "[ERR] Mismatch on distances: "
                    "vid = {}, cugraph = {}, nx = {}".format(
                        vertex, result, expected
                    )
                )
                distance_mismatch_error += 1
            if vertex not in cu_predecessors:
                missing_vertex_error += 1
            else:
                pred = cu_predecessors[vertex]
                if vertex != source and pred not in nx_distances:
                    invalid_predecessor_error += 1
                else:
                    # The graph is unweighted thus, predecessors are 1 away
                    if vertex != source and (
                        (nx_distances[pred] + 1 != cu_distances[vertex])
                    ):
                        print(
                            "[ERR] Invalid on predecessors: "
                            "vid = {}, cugraph = {}".format(vertex, pred)
                        )
                        invalid_predecessor_error += 1
        else:
            missing_vertex_error += 1
    assert missing_vertex_error == 0, "There are missing vertices"
    assert distance_mismatch_error == 0, "There are invalid distances"
    assert invalid_predecessor_error == 0, "There are invalid predecessors"


def _compare_bfs_with_spc(cugraph_df, nx_sp_counter):
    """
    Compare BFS with shortest path counters.
    """
    sorted_nx = [nx_sp_counter[key] for key in sorted(nx_sp_counter.keys())]

    # This does not check for distances / predecessors here as it is assumed
    # that those have been checked in the _compare_bfs tests, this focuses
    # solely on shortest path counting.

    # cugraph returns a dataframe that should contain each vertex without
    # repeats.  This could be used to filter only vertices that are common to
    # both, but it would slow down the comparison, and in this specific case
    # nxacb._single_source_shortest_path_basic is a dictionary containing all
    # the vertices.  There is no guarantee that the vertices in cugraph_df are
    # sorted so order is enforced to enable faster comparisons.
    sorted_df = cugraph_df.sort_values("vertex").rename(
        columns={"sp_counter": "cu_spc"}, copy=False
    )

    # This allows to detect vertices identifier that could have been
    # wrongly present multiple times
    cu_vertices = set(sorted_df['vertex'].values_host)
    nx_vertices = nx_sp_counter.keys()
    assert len(cu_vertices.intersection(nx_vertices)) == len(
        nx_vertices
    ), "There are missing vertices"

    # nx shortest path counter is added to the cudf.DataFrame, so both the
    # the DataFrame and `sorted_nx` are sorted base on vertex identifiers.
    sorted_df["nx_spc"] = sorted_nx

    # numpy.isclose or cupy.isclose could be used to get the mismatches in the
    # cudf.DataFrame entries.  numpy / cupy allclose would get only a boolean
    # and the extra information about the discrepancies might be needed.
    shortest_path_counter_errors = sorted_df[
        ~cupy.isclose(
            sorted_df["cu_spc"], sorted_df["nx_spc"], rtol=_default_epsilon
        )
    ]
    if len(shortest_path_counter_errors) > 0:
        print(shortest_path_counter_errors)
    assert len(shortest_path_counter_errors) == 0, (
        "Shortest path counters " "are too different"
    )


################################################################################
## BFS-specific fixtures

@pytest.fixture(scope="module")
def cugraphobj_nxresults_startvertex(cugraph_nx_objs_from_dataset):
    """
    Returns the tuple of (cugraph obj, NetworkX BFS results, random startvertex
    used) for each item in cugraph_nx_obj_from_dataset.
    """
    (dataset_obj, df, G, Gnx) = cugraph_nx_objs_from_dataset

    random.seed(_seed)
    start_vertex = random.sample(Gnx.nodes(), 1)[0]
    nx_values = nx.single_source_shortest_path_length(Gnx, start_vertex)

    return (G, nx_values, start_vertex)


@pytest.fixture(scope="module")
def cugraphobj_nxresults_allstartvertices(cugraph_nx_objs_from_dataset):
    """
    Returns the tuple of (cugraph obj, NetworkX BFS all paths results, all start
    vertices) for each item in cugraph_nx_obj_from_dataset.
    """
    (dataset_obj, df, G, Gnx) = cugraph_nx_objs_from_dataset

    start_vertices = [start_vertex for start_vertex in Gnx]

    all_nx_values = []
    for start_vertex in start_vertices:
        _, _, nx_sp_counter = \
            nxacb._single_source_shortest_path_basic(Gnx, start_vertex)
        nx_values = nx_sp_counter
        all_nx_values.append(nx_values)

    return (G, all_nx_values, start_vertices)


################################################################################
## Tests

def test_bfs(gpubenchmark, cugraphobj_nxresults_startvertex):
    """
    Test BFS traversal on random source with distance and predecessors
    """
    (G, nx_values, start_vertex) = cugraphobj_nxresults_startvertex

    cu_values = gpubenchmark(cugraph.bfs_edges, G, start_vertex)

    _compare_bfs(cu_values, nx_values, start_vertex)


def test_bfs_spc_full(gpubenchmark, cugraphobj_nxresults_allstartvertices):
    """
    Test BFS traversal on every vertex with shortest path counting
    """
    (G, all_nx_values, start_vertices) = cugraphobj_nxresults_allstartvertices

    all_cugraph_values = []
    def func_to_benchmark():
        for sv in start_vertices:
            cugraph_df = cugraph.bfs_edges(G, sv, return_sp_counter=True)
            all_cugraph_values.append(cugraph_df)

    gpubenchmark(func_to_benchmark)

    for i in range(len(start_vertices)):
        cugraph_df = all_cugraph_values[i]
        assert len(cugraph_df.columns) == 4, \
               "The result of the BFS has an invalid number of columns"
        _compare_bfs_with_spc(cugraph_df, all_nx_values[i])


# @pytest.mark.parametrize("cugraph_input_type",
#                          utils.NX_INPUT_TYPES + utils.MATRIX_INPUT_TYPES)
# def test_bfs_nonnative_inputs(gpubenchmark,
#                               single_dataset_nxresults_startvertex_spc,
#                               cugraph_input_type):
#     test_bfs(gpubenchmark,
#              single_dataset_nxresults_startvertex_spc,
#              cugraph_input_type)
#
#
# def test_scipy_api_compat():
#     graph_file = utils.DATASETS[0]
#
#     input_cugraph_graph = utils.create_obj_from_csv(graph_file, cugraph.Graph,
#                                                     edgevals=True)
#     input_coo_matrix = utils.create_obj_from_csv(graph_file, cp_coo_matrix,
#                                                  edgevals=True)
#     # Ensure scipy-only options are rejected for cugraph inputs
#     with pytest.raises(TypeError):
#         cugraph.bfs(input_cugraph_graph, start=0, directed=False)
#     with pytest.raises(TypeError):
#         cugraph.bfs(input_cugraph_graph)  # required arg missing
#
#     # Ensure cugraph-compatible options work as expected
#     cugraph.bfs(input_cugraph_graph, i_start=0)
#     cugraph.bfs(input_cugraph_graph, i_start=0, return_sp_counter=True)
#     # cannot have start and i_start
#     with pytest.raises(TypeError):
#         cugraph.bfs(input_cugraph_graph, start=0, i_start=0)
#
#     # Ensure SciPy options for matrix inputs work as expected
#     cugraph.bfs(input_coo_matrix, i_start=0)
#     cugraph.bfs(input_coo_matrix, i_start=0, directed=True)
#     cugraph.bfs(input_coo_matrix, i_start=0, directed=False)
#     result = cugraph.bfs(input_coo_matrix, i_start=0,
#                          return_sp_counter=True)
#     assert type(result) is tuple
#     assert len(result) == 3
