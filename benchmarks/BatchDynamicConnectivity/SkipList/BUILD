cc_library(
  name = "SkipList",
  hdrs = ["SkipList.h"],
  deps = [
  "//gbbs:julienne",
  "//benchmarks/KCore/JulienneDBS17:KCore_main",
  "//gbbs/helpers:sparse_table",
  "//gbbs:graph_io",
  "//gbbs:dynamic_graph_io"
  ]
)

cc_binary(
  name = "SkipList_main",
  srcs = ["SkipList.cc"],
  deps = [":SkipList"]
)

package(
  default_visibility = ["//visibility:public"],
)
