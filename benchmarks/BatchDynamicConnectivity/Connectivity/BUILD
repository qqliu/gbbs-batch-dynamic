cc_library(
  name = "Connectivity",
  hdrs = ["Connectivity.h"],
  deps = [
  "//gbbs:gbbs",
  "//gbbs/helpers:sparse_table",
  "//gbbs:julienne",
  "//benchmarks/KCore/JulienneDBS17:KCore_main",
  "//benchmarks/BatchDynamicConnectivity/EulerTourTree:ETTree_main",
  "//benchmarks/SpanningForest/SDB14:SpanningForest_main",
  "//gbbs:graph_io",
  "//gbbs:dynamic_graph_io"
  ]
)

cc_binary(
  name = "Connectivity_main",
  srcs = ["Connectivity.cc"],
  deps = [":Connectivity"]
)

package(
  default_visibility = ["//visibility:public"],
)
