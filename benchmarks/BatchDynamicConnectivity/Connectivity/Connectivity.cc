// Usage (dynamic):
// numactl -i all ./LDS -rounds 3 -s -eps 0.4 -delta 3 -i <dynamic graph> -b 1000 <static graph>
// flags:
//   required:
//     -s : indicates that the graph is symmetric
//   optional:
//     -m : indicate that the graph should be mmap'd
//     -c : indicate that the graph is compressed
//     -rounds : the number of times to run the algorithm
//     -i : file path to the dynamic graph, with a single dynamic edge on each
//          line, in the format "<+/-> u v", where + denotes insertion, -
//          denotes deletion, and (u, v) are the vertex ids denoting the edge
//          (e.g., "+ 0 1")
//     -b : batch size (output includes time per batch, from start_size to
//          end_size)
//     -eps: epsilon
//     -delta: lambda
//     -stats : indicates whether to output comparisons to exact coreness
//              values
//     -ins-opt : indicates whether to set lambda such that
//                (2 + 3 / lambda) = 1.1
//     -size : option for getting the size of the data structure in bytes
//     -opt : indicates whether to divide the number of levels by 50; faster
//     runtime, subtle decrease in accuracy
//     -init_graph_file: initial graph for which the additional edge updates
//     occur on; initial graph should follow the same format as the input to -i
//     (i.e. all edges should be prepended with '+')
// Note: The static graph is ignored if -i is specified.

#include "Connectivity.h"

namespace gbbs {
template <class Graph>
double Connectivity_runner(Graph& G, commandLine P) {
  std::cout << "started connectivity runner" << std::endl;
  timer t; t.start();
  double tt = t.stop();
  using W = typename Graph::weight_type;
  const std::string kInputFlag{"-i"};
  const char* const input_file{P.getOptionValue(kInputFlag)};
  bool compare_exact = P.getOption("-stats");

  std::cout << "reading edge list" << std::endl;
  bool use_dynamic = (input_file && input_file[0]);
  BatchDynamicEdges<W> batch_edge_list = use_dynamic ?
    read_batch_dynamic_edge_list<W>(input_file) : BatchDynamicEdges<W>{};

  // Options not exposed to general users
  const char* const init_graph_file(P.getOptionValue("-init_graph_file"));

  // Prepend the initial graph if specified, and store the offset of the
  // initial graph in the resulting dynamic edges list
  size_t offset = 0;
  if (use_dynamic && init_graph_file) {
    BatchDynamicEdges<W> init_graph_list =
      read_batch_dynamic_edge_list<W>(init_graph_file);
    offset = prepend_dynamic_edge_list(batch_edge_list, init_graph_list);
  }

  std::cout << "finished reading edge list" << std::endl;
  //RunConnectivity(batch_edge_list, 5, false, 0, 1140148, 50, 2787967, 1.5);
  RunConnectivity(batch_edge_list, 10000, compare_exact, offset, 7115, 20, 1.01);

  std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}
}  // namespace gbbs

generate_symmetric_main(gbbs::Connectivity_runner, false);
