#include <unordered_set>
#include <stack>
#include <limits>

#include "gbbs/gbbs.h"
#include "gbbs/dynamic_graph_io.h"
#include "gbbs/helpers/sparse_table.h"
#include "benchmarks/KCore/JulienneDBS17/KCore.h"
#include "benchmarks/BatchDynamicConnectivity/EulerTourTree/ETTree.h"

namespace gbbs {
using K = std::pair<uintE, uintE>;
using V = uintE;
using KV = std::pair<K, V>;


// A union-find node that stores its parent and rank
struct UFNode {
  uintE parent;
  uintE rank;
  UFNode() {}

  UFNode(int p, int r) : parent(p), rank(r) {}
};

// A union-find data structure that supports parallel find and union operations
class UnionFind {
  public:
  // An array of union-find nodes
  sequence<UFNode> nodes;

  UnionFind() {}
  // Constructor that initializes each node as a singleton set
  UnionFind(size_t n) {
      nodes = sequence<UFNode>(n);
      parallel_for(0, n, [&] (size_t i) {
        nodes[i] = UFNode(i, 0);
      });
  }

  // Find the root of the set containing x
  uintE find(uintE x) {
    // Use path compression to update the parent pointers along the path
    if (nodes[x].parent != x) {
      nodes[x].parent = find(nodes[x].parent);
    }
    return nodes[x].parent;
  }

  // Union the sets containing x and y
  void unionf(uintE x, uintE y) {
    // Find the roots of x and y
    int xRoot = find(x);
    int yRoot = find(y);
    // If they are already in the same set, do nothing
    if (xRoot == yRoot) return;
    // Use union by rank to merge the smaller set into the larger one
    if (nodes[xRoot].rank < nodes[yRoot].rank) {
      nodes[xRoot].parent = yRoot;
    } else if (nodes[xRoot].rank > nodes[yRoot].rank) {
      nodes[yRoot].parent = xRoot;
    } else {
      nodes[yRoot].parent = xRoot;
      nodes[xRoot].rank++;
    }
  }

  // Parallel find operation that returns an array of roots for each element
  parlay::sequence<std::pair<uintE, uintE>> parallel_find(sequence<std::pair<uintE, uintE>> nodes) {
    // Use parallel reduce to find the roots of each element in parallel
    sequence<std::pair<uintE, uintE>> parents = sequence<std::pair<uintE, uintE>>(nodes.size());
    parallel_for(0, nodes.size(),  [&](size_t i) {
      parents[i] = std::make_pair(find(nodes[i].first), find(nodes[i].second));
    });

    return parents;
  }
};

// A type alias for an edge represented by a pair of uintE
using edge_mst_type = std::pair<uintE, uintE>;

// A function that returns a sequence of edges in the spanning forest
sequence<edge_mst_type> parallel_spanning_forest(sequence<edge_mst_type> edges, size_t n) {
  // Initialize a union-find data structure
  UnionFind uf(n);
  // Initialize a sequence of edges in the spanning forest
  sequence<edge_mst_type> forest;
  // Repeat until all the edges are processed
  while (!edges.empty()) {
    // Find the roots of each endpoint in parallel
    auto not_selected = sequence<bool>(edges.size(), true);

    auto endpoints = sequence<std::pair<uintE, uintE>>(edges.size());
    parallel_for(0, edges.size(), [&](size_t i){
        endpoints[i] = std::make_pair(edges[i].first, edges[i].second);
    });

    auto roots = uf.parallel_find(endpoints);

    using K = std::pair<uintE, uintE>;
    using V = std::pair<std::pair<uintE, uintE>, uintE>;
    using KV = std::pair<K, V>;

    KV empty =
        std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), std::make_pair(
                std::make_pair(UINT_E_MAX, UINT_E_MAX), UINT_E_MAX));

    auto hash_pair = [](const std::pair<uintE, uintE>& t) {
        size_t l = std::min(std::get<0>(t), std::get<1>(t));
        size_t r = std::max(std::get<0>(t), std::get<1>(t));
        size_t key = (l << 32) + r;
        return parlay::hash64_2(key);
    };

    auto root_edge_to_original =
         gbbs::make_sparse_table<K, V>(roots.size(), empty, hash_pair);

    parallel_for(0, roots.size(), [&](size_t i){
        auto edge = roots[i];
        auto u = std::get<0>(edge);
        auto v = std::get<1>(edge);
        //std::cout << "root edges: " << u << ", " << v << std::endl;

        bool abort = false;
        root_edge_to_original.insert_check(std::make_pair(std::make_pair(u,
            v), std::make_pair(endpoints[i], i)), &abort);
    });

    // Filter out the edges that connect vertices in the same component

    auto bool_filter = sequence<bool>(roots.size());
    parallel_for(0, bool_filter.size(), [&](size_t i){
        bool_filter[i] = roots[i].first != roots[i].second;
        not_selected[i] = bool_filter[i];
    });
    auto filtered = parlay::pack(roots, bool_filter);

    auto compare_tup = [&] (const std::pair<uintE, uintE> l, const std::pair<uintE, uintE> r) {
        return l.first < r.first;
    };
    parlay::sort_inplace(parlay::make_slice(filtered), compare_tup);

    auto bool_seq = sequence<bool>(filtered.size() + 1);
    parallel_for(0, filtered.size() + 1, [&](size_t i) {
        bool_seq[i] = (i == 0) || (i == filtered.size()) ||
            (filtered[i-1].first != filtered[i].first);
    });
    auto starts = parlay::pack_index(bool_seq);

    auto selected = sequence<edge_mst_type>(starts.size()-1);

    parallel_for(0, starts.size()-1, [&](size_t i){
        auto original_edge = root_edge_to_original.find(filtered[starts[i]],
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), UINT_E_MAX));
        selected[i] = original_edge.first;
        not_selected[original_edge.second] = false;
    });

    // Append the selected edges to the forest
    forest.append(selected);

    // Union the components connected by the selected edges
    parallel_for(0, selected.size(), [&](size_t i) {
      uf.unionf(selected[i].first, selected[i].second);
    });

    // Remove the selected edges from the original sequence
    edges = parlay::pack(edges, not_selected);
  }
  // Return the forest
  return forest;
}

struct Connectivity {
    size_t n;
    ETTree tree;

    Connectivity() {};

    Connectivity(size_t n_, int copies_, size_t m_, double pb_) {
        n = n_;
        tree = ETTree(n_, copies_, m_, pb_);
    }

    // initialize edgemap for SkipListElement data structures
    template <class KY, class VL, class HH, class W>
    void initialize_data_structures(BatchDynamicEdges<W>& batch_edge_list,
            gbbs::sparse_table<KY, VL, HH> edge_table) {
        auto all_edges = batch_edge_list.edges;

        bool abort = false;

        parallel_for(0, all_edges.size(), [&](size_t i){
            uintE v = all_edges[i].from;
            uintE w = all_edges[i].to;

            if (all_edges[i].insert) {
                edge_table.insert_check(std::make_pair(std::make_pair(v, w), 2 * i), &abort);
                edge_table.insert_check(std::make_pair(std::make_pair(v, w), 2 * i + 1), &abort);
            }
        });
    }

    template <class Seq, class KY, class VL, class HH>
    void batch_insertion(const Seq& insertions, gbbs::sparse_table<KY, VL, HH> edge_table) {
        auto non_empty_spanning_tree = true;
        auto first = true;
        sequence<std::pair<uintE, uintE>> edges_both_directions = sequence<std::pair<uintE, uintE>>(0);
        sequence<size_t> starts = sequence<size_t>(0);
        sequence<SkipList::SkipListElement*> representative_nodes = sequence<SkipList::SkipListElement*>(0);

        std::cout << "next batch" << std::endl;
        while(non_empty_spanning_tree) {
            std::cout << "next spanning tree" << std::endl;
            //std::cout << "number of edges in insertions: " << insertions.size() << std::endl;
            if (first) {
                edges_both_directions =  sequence<std::pair<uintE, uintE>>(2 * insertions.size());

                parallel_for(0, insertions.size(), [&](size_t i) {
                    //std::cout << "original edge: " << insertions[i].first << ", " << insertions[i].second << std::endl;
                    auto [u, v] = insertions[i];
                    //tree.create_edge_in_edge_table(u, v);
                    edges_both_directions[2 * i] = std::make_pair(u, v);
                    edges_both_directions[2 * i + 1] = std::make_pair(v, u);
                });

                auto compare_tup = [&] (const std::pair<uintE, uintE> l, const std::pair<uintE, uintE> r) {
                    return l.first < r.first;
                };
                parlay::sort_inplace(parlay::make_slice(edges_both_directions), compare_tup);

                auto bool_seq = sequence<bool>(edges_both_directions.size() + 1);
                parallel_for(0, edges_both_directions.size() + 1, [&](size_t i) {
                    bool_seq[i] = (i == 0) || (i == edges_both_directions.size()) ||
                       (edges_both_directions[i-1].first != edges_both_directions[i].first);
                });
                starts = parlay::pack_index(bool_seq);

                representative_nodes =
                    sequence<SkipList::SkipListElement*>(starts.size()-1);
                sequence<SkipList::SkipListElement*> nodes_to_update =
                    sequence<SkipList::SkipListElement*>(starts.size()-1);
                auto update_seq =
                    sequence<std::pair<SkipList::SkipListElement*,
                    sequence<sequence<std::pair<uintE, uintE>>>>>(starts.size()- 1);

            /*std::cout << "starts size: " << starts.size() << std::endl;
            std::cout << "edges both directions size: " << edges_both_directions.size() << std::endl;
            for (int i = 0; i < starts.size(); i++) {
                    std::cout << "starts: " << starts[i] << std::endl;
            }

            for (int i = 0; i < edges_both_directions.size(); i++) {
                    std::cout << "edge: " << edges_both_directions[i].first << ", " << edges_both_directions[i].second
                        << std::endl;
            }*/

                parallel_for(0, starts.size() - 1, [&](size_t i) {
                    for (size_t j = starts[i]; j < starts[i+1]; j++) {
                    /*std::cout << "starts for " << i <<
                        ": " << edges_both_directions[j].first << ", "
                        << edges_both_directions[j].second << std::endl;*/

                        tree.add_edge_to_cutsets(edges_both_directions[j]);
                    }
                    SkipList::SkipListElement* our_vertex = &tree.vertices[edges_both_directions[starts[i]].first];
                    representative_nodes[i] =
                        tree.skip_list.find_representative(our_vertex);
                    update_seq[i] = std::make_pair(our_vertex, our_vertex->values[0]);
                });
                tree.skip_list.batch_update_xor(&update_seq);
            }
            first = false;
            if (!first) {
                parallel_for(0, starts.size() - 1, [&](size_t i) {
                    std::cout << "printing values" << std::endl;
                    std::cout << "new starts: " << starts[i] << ", " << edges_both_directions[starts[i]].first
                        << std::endl;
                    std::cout << "rep nodes size: " << representative_nodes.size() << std::endl;
                    SkipList::SkipListElement* our_vertex = &tree.vertices[edges_both_directions[starts[i]].first];
                    std::cout << "got list element: " << representative_nodes.size() << std::endl;
                    representative_nodes[i] =
                        tree.skip_list.find_representative(our_vertex);
                    std::cout << "finished updating representative" << std::endl;
                });
            }

            std::cout << "doing update" << std::endl;

            sequence<std::pair<uintE, uintE>> found_possible_edges =
                    sequence<std::pair<uintE, uintE>>(representative_nodes.size());
            sequence<bool> is_edge = sequence<bool>(representative_nodes.size());

            parallel_for(0, representative_nodes.size(), [&](size_t i) {
                auto id = representative_nodes[i]->id.first;
                auto sum = tree.get_tree_xor(id);
                bool is_present = false;
                bool is_real_edge = false;

                for (size_t ii = 0; ii < sum.size(); ii++) {
                if (!is_real_edge) {
                    for (size_t ij = 0; ij < sum[ii].size(); ij++) {
                        if (!is_real_edge) {
                            //std::cout << "rep values for " << id << ": " << sum[ii][ij].first << ", " << sum[ii][ij].second
                                //<< std::endl;
                            is_present = sum[ii][ij].first != UINT_E_MAX && sum[ii][ij].second != UINT_E_MAX;
                            auto u = sum[ii][ij].first;
                            auto v = sum[ii][ij].second;
                            if (is_present) {
                                if (u < n && v < n) {
                                    auto index = edge_table.find(std::make_pair(u, v), UINT_E_MAX); //, UINT_E_MAX);
                                    //auto element = &tree.edge_table[u][v];
                                    /*std::cout << "element: " << element->id.first << ", " << element->id.second <<
                                        std::endl;*/

                                    if (index != UINT_E_MAX) { // && element->id.second != UINT_E_MAX) {
                                        is_real_edge = true;
                                        found_possible_edges[i] = std::make_pair(u, v);
                                    }
                                }
                            }
                        }
                    }
                }
                }

                /*auto possible_edge =
                    tree.edge_table[found_possible_edges[i].first][found_possible_edges[i].second];*/
                is_edge[i] = is_real_edge;
            });

            auto real_edges = parlay::pack(found_possible_edges, is_edge);
            auto representative_edges = sequence<std::pair<std::pair<uintE, uintE>, uintE>>(real_edges.size());

            size_t max_index = 0;
            parallel_for(0, real_edges.size(), [&](size_t i) {
                auto u = tree.vertices[real_edges[i].first];
                auto v = tree.vertices[real_edges[i].second];
                auto node_a = tree.skip_list.find_representative(&u);
                auto node_b = tree.skip_list.find_representative(&v);
                auto node_a_id = std::min(node_a->id.first, node_a->id.second);
                auto node_b_id = std::min(node_b->id.first, node_b->id.second);
                if (node_a_id > max_index)
                    max_index = node_a_id;
                if (node_b_id > max_index)
                    max_index = node_b_id;

                auto ru = std::min(node_a_id, node_b_id);
                auto rv = std::max(node_a_id, node_b_id);

                representative_edges[i] = std::make_pair(std::make_pair(ru, rv), i);
            });

            auto compare_tup = [&] (const std::pair<std::pair<uintE, uintE>, uintE> l,
                    const std::pair<std::pair<uintE, uintE>,
                    uintE> r) {
                return (l.first.first == r.first.first && l.first.second <= r.first.second)
                    || (l.first.first < r.first.first);
            };
            parlay::sort_inplace(parlay::make_slice(representative_edges), compare_tup);

            auto bool_seq = sequence<bool>(representative_edges.size() + 1);
            parallel_for(0, representative_edges.size() + 1, [&](size_t i) {
            bool_seq[i] = (i == 0) || (i == representative_edges.size()) ||
                ((representative_edges[i-1].first != representative_edges[i].first) ||
                 (representative_edges[i-1].second != representative_edges[i].second));
            });
            auto starts = parlay::pack_index(bool_seq);
            auto unique_representative_edges = sequence<std::pair<uintE, uintE>>(starts.size());
            auto unique_real_edges = sequence<std::pair<uintE, uintE>>(starts.size());

            parallel_for(0, starts.size() - 1, [&](size_t i){
                unique_representative_edges[i] = representative_edges[starts[i]].first;
                auto unique_edge = real_edges[representative_edges[starts[i]].second];
                std::cout << "Unique edge: " << unique_edge.first << ", " << unique_edge.second << std::endl;
                unique_real_edges[i] = unique_edge;
            });

            using K = std::pair<uintE, uintE>;
            using V = std::pair<uintE, uintE>;
            using KV = std::pair<K, V>;

            KV empty =
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), std::make_pair(UINT_E_MAX, UINT_E_MAX));
            auto hash_pair = [](const std::pair<uintE, uintE>& t) {
                    size_t l = std::min(std::get<0>(t), std::get<1>(t));
                    size_t r = std::max(std::get<0>(t), std::get<1>(t));
                    size_t key = (l << 32) + r;
                return parlay::hash64_2(key);
            };
            auto representative_edge_to_original =
                gbbs::make_sparse_table<K, V>(unique_representative_edges.size(), empty, hash_pair);

            parallel_for(0, unique_representative_edges.size(), [&](size_t i){
                auto edge = unique_representative_edges[i];
                auto vert1 = std::get<0>(edge);
                auto vert2 = std::get<1>(edge);
                auto u = std::min(vert1, vert2);
                auto v = std::max(vert1, vert2);
                //std::cout << "rep edges: " << u << ", " << v << std::endl;

                bool abort = false;
                representative_edge_to_original.insert_check(std::make_pair(std::make_pair(u,
                        v), unique_real_edges[i]), &abort);
            });

            auto spanning_forest = parallel_spanning_forest(unique_representative_edges, n);
            if(spanning_forest.size() == 0)
                non_empty_spanning_tree = false;

            auto original_edges = sequence<std::pair<uintE, uintE>>(spanning_forest.size());
            parallel_for(0, spanning_forest.size(), [&](size_t i) {
                auto vert1 = std::get<0>(spanning_forest[i]);
                auto vert2 = std::get<1>(spanning_forest[i]);
                std::cout << "spanning forest edge: " << vert1 << ", " << vert2 << std::endl;

                auto u = std::min(vert1, vert2);
                auto v = std::max(vert1, vert2);
                //std::cout << "spanning tree edges: " << u << ", " << v << std::endl;

                auto original_edge = representative_edge_to_original.find(std::make_pair(u, v), std::make_pair(UINT_E_MAX, UINT_E_MAX));
                std::cout << "found original edges: "<< original_edge.first << ", " << original_edge.second << std::endl;
                if (original_edge.first == UINT_E_MAX || original_edge.second == UINT_E_MAX)
                    std::cout << "FAILURE: THERE IS A BUG" << std::endl;
                original_edges[i] = std::make_pair(std::min(original_edge.first, original_edge.second),
                    std::max(original_edge.first, original_edge.second));
                std::cout << "checking original edges: " << original_edges[i].first << ", " <<
                    original_edges[i].second << std::endl;
            });

            auto compare_tup1 = [&] (const std::pair<uintE, uintE> l, const std::pair<uintE, uintE> r) {
                return (l.first == r.first && l.second <= r.second) || (l.first < r.first);
            };
            parlay::sort_inplace(parlay::make_slice(original_edges), compare_tup1);

            auto bool_seq1 = sequence<bool>(original_edges.size() + 1);
            parallel_for(0, original_edges.size() + 1, [&](size_t i) {
                bool_seq1[i] = (i == 0) || (i == original_edges.size()) ||
                    ((original_edges[i-1].first != original_edges[i].first) || (original_edges[i-1].second !=
                     original_edges[i].second));
            });
            auto starts1 = parlay::pack_index(bool_seq1);
            std::cout << "starts size " << starts1.size() << ": ";
            for (size_t i = 0; i < starts1.size(); i++)
                std::cout << starts1[i];
            std::cout << std::endl;
            auto unique_original_edges = sequence<std::pair<uintE, uintE>>(starts1.size()-1);

            parallel_for(0, starts1.size() - 1, [&](size_t i){
                unique_original_edges[i] = original_edges[starts1[i]];
                std::cout << "Unique original edge: " << unique_original_edges[i].first
                    << ", " << unique_original_edges[i].second << std::endl;
            });

            tree.batch_link(unique_original_edges, edge_table);
            std::cout << "end current spanning tree" << std::endl;
            //non_empty_spanning_tree = false;
         }
    }

    bool is_connected(uintE u, uintE v) {
            auto uvert = &tree.vertices[u];
            auto vvert = &tree.vertices[v];
            return tree.skip_list.find_representative(uvert)
                == tree.skip_list.find_representative(vvert);
    }

    template <class Seq, class KY, class VL, class HH>
    void batch_deletion(const Seq& deletions, gbbs::sparse_table<KY, VL, HH> edge_table) {
    }
};

void RunConnectivityTest() {
        std::cout << "Connectivity Test" << std::endl;
}

template <class W>
inline void RunConnectivity(BatchDynamicEdges<W>& batch_edge_list, long batch_size, bool compare_exact,
        size_t offset, size_t n, int copies, size_t m, double pb) {
        auto cutset = Connectivity(n, copies, m, pb);
        auto batch = batch_edge_list.edges;

        KV empty =
            std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), UINT_E_MAX);

        auto hash_pair = [](const std::pair<uintE, uintE>& t) {
            size_t l = std::min(std::get<0>(t), std::get<1>(t));
            size_t r = std::max(std::get<0>(t), std::get<1>(t));
            size_t key = (l << 32) + r;
            return parlay::hash64_2(key);
        };

        auto edge_table =
            gbbs::make_sparse_table<K, V>(2 * batch.size(), empty, hash_pair);

        auto existence_table =
            gbbs::make_sparse_table<K, bool>(2 * batch.size(), empty, hash_pair);

        cutset.initialize_data_structures(batch_edge_list, edge_table);
        bool abort = false;

        std::cout << "batch size: " << batch.size() << std::endl;

        if (offset != 0) {
            for (size_t i = 0; i < offset; i += 1000000) {
                auto end_size = std::min(i + 1000000, offset);
                auto insertions = parlay::filter(parlay::make_slice(batch.begin() + i,
                        batch.begin() + end_size), [&] (const DynamicEdge<W>& edge){
                    return edge.insert;
                });
                auto deletions = parlay::filter(parlay::make_slice(batch.begin() + i,
                        batch.begin() + end_size), [&] (const DynamicEdge<W>& edge){
                    return !edge.insert;
                });
                auto batch_insertions = parlay::delayed_seq<std::pair<uintE, uintE>>(insertions.size(),
                    [&] (size_t i) {
                    uintE vert1 = insertions[i].from;
                    uintE vert2 = insertions[i].to;

                    existence_table.insert_check(std::make_pair(std::make_pair(vert1,
                        vert2), true), &abort);
                    existence_table.insert_check(std::make_pair(std::make_pair(vert2,
                        vert1), true), &abort);

                    return std::make_pair(vert1, vert2);
                });

                auto batch_deletions = parlay::delayed_seq<std::pair<uintE, uintE>>(deletions.size(),
                [&] (size_t i) {
                    uintE vert1 = deletions[i].from;
                    uintE vert2 = deletions[i].to;

                    existence_table.insert_check(std::make_pair(std::make_pair(vert1,
                        vert2), false), &abort);
                    existence_table.insert_check(std::make_pair(std::make_pair(vert2,
                        vert1), false), &abort);

                    return std::make_pair(vert1, vert2);
                });

                cutset.batch_insertion(batch_insertions, edge_table);
                cutset.batch_deletion(batch_deletions, edge_table);
            }
        }

        for (size_t i = offset; i < batch.size(); i += batch_size) {
            std::cout << "batch: " << i << std::endl;
            timer t; t.start();
            auto end_size = std::min(i + batch_size, batch.size());
            auto insertions = parlay::filter(parlay::make_slice(batch.begin() + i,
                    batch.begin() + end_size), [&] (const DynamicEdge<W>& edge){
                return edge.insert;
            });

            auto deletions = parlay::filter(parlay::make_slice(batch.begin() + i,
                    batch.begin() + end_size), [&] (const DynamicEdge<W>& edge){
                return !edge.insert;
            });

            auto batch_insertions = parlay::delayed_seq<std::pair<uintE, uintE>>(insertions.size(),
                [&] (size_t i) {
                uintE vert1 = insertions[i].from;
                uintE vert2 = insertions[i].to;

                existence_table.insert_check(std::make_pair(std::make_pair(vert1,
                    vert2), true), &abort);
                existence_table.insert_check(std::make_pair(std::make_pair(vert2,
                    vert1), true), &abort);

                return std::make_pair(vert1, vert2);
            });


            auto batch_deletions = parlay::delayed_seq<std::pair<uintE, uintE>>(deletions.size(),
                [&] (size_t i) {
                uintE vert1 = deletions[i].from;
                uintE vert2 = deletions[i].to;

                existence_table.insert_check(std::make_pair(std::make_pair(vert1,
                    vert2), false), &abort);
                existence_table.insert_check(std::make_pair(std::make_pair(vert2,
                    vert1), false), &abort);

                return std::make_pair(vert1, vert2);
            });

            cutset.batch_insertion(batch_insertions, edge_table);
            cutset.batch_deletion(batch_deletions, edge_table);

            sequence<int> correct = sequence<int>(batch_insertions.size(), false);
             parallel_for(0, batch_insertions.size(), [&](size_t i) {
                correct[i] = cutset.is_connected(batch_insertions[i].first, batch_insertions[i].second);
                //std::cout << "CONNECTED: " << correct[i] << std::endl;
            });
            auto num_correct = parlay::scan_inplace(correct);

            std::cout << "fraction correct: " << (num_correct * 1.0)/batch_insertions.size() << std::endl;
    }
}

}  // namespace gbbs
