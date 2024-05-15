cc_library(
  name = "ETTree",
  hdrs = ["ETTree.h"],
  deps = [
  "//gbbs:gbbs",
  "//gbbs/helpers:sparse_table",
  "//gbbs:julienne",
  "//benchmarks/KCore/JulienneDBS17:KCore_main",
  "//benchmarks/BatchDynamicConnectivity/SkipList:SkipList_main",
  "//gbbs:graph_io",
  "//gbbs:dynamic_graph_io"
  ]
)

cc_binary(
  name = "ETTree_main",
  srcs = ["ETTree.cc"],
  deps = [":ETTree"]
)

package(
  default_visibility = ["//visibility:public"],
)
