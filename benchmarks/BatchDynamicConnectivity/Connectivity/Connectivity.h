#include <unordered_set>
#include <stack>
#include <limits>

#include "gbbs/gbbs.h"
#include "gbbs/dynamic_graph_io.h"
#include "gbbs/helpers/sparse_table.h"
#include "benchmarks/KCore/JulienneDBS17/KCore.h"
#include "benchmarks/BatchDynamicConnectivity/EulerTourTree/ETTree.h"
#include "benchmarks/SpanningForest/SDB14/SpanningForest.h"

namespace gbbs {
using K = std::pair<uintE, uintE>;
using V = uintE;
using KV = std::pair<K, V>;

using V_pairs = std::pair<uintE, uintE>;
using KV_pairs = std::pair<K, V_pairs>;

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
            gbbs::sparse_table<KY, VL, HH>& edge_table) {
        auto all_edges = batch_edge_list.edges;

        bool abort = false;

        parallel_for(0, all_edges.size(), [&](size_t i){
            uintE v = all_edges[i].from;
            uintE w = all_edges[i].to;

            if (all_edges[i].insert) {
                edge_table.insert_check(std::make_pair(std::make_pair(v, w), 2 * i), &abort);
                edge_table.insert_check(std::make_pair(std::make_pair(w, v), 2 * i + 1), &abort);
            }
        });
    }

    template <class Seq, class KY, class VL, class HH>
    void batch_insertion(const Seq& insertions, gbbs::sparse_table<KY, VL, HH>& edge_table,
            gbbs::sparse_table<KY, bool, HH>& existence_table) {

        auto non_empty_spanning_tree = true;
        auto first = true;
        sequence<std::pair<uintE, uintE>> edges_both_directions = sequence<std::pair<uintE, uintE>>(0);
        sequence<size_t> starts = sequence<size_t>(0);
        sequence<SkipList::SkipListElement*> representative_nodes = sequence<SkipList::SkipListElement*>(0);
        bool abort = false;

        while(non_empty_spanning_tree) {
            if (first) {
                edges_both_directions =  sequence<std::pair<uintE, uintE>>(2 * insertions.size());

                parallel_for(0, insertions.size(), [&](size_t i) {
                    auto [u, v] = insertions[i];
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

                parallel_for(0, starts.size() - 1, [&](size_t i) {
                    // update j's cutset data structure; need to be sequential because accessing same arrays
                    for (size_t j = starts[i]; j < starts[i+1]; j++) {
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
                    SkipList::SkipListElement* our_vertex = &tree.vertices[edges_both_directions[starts[i]].first];
                    representative_nodes[i] =
                        tree.skip_list.find_representative(our_vertex);
                });
            }

            parlay::sort_inplace(parlay::make_slice(representative_nodes));

            auto bool_seq = sequence<bool>(representative_nodes.size());
            parallel_for(0, representative_nodes.size(), [&](size_t i) {
                bool_seq[i] = (i == 0) ||
                       (edges_both_directions[i-1] != edges_both_directions[i]);
            });
            auto representative_starts = parlay::pack_index(bool_seq);

            sequence<std::pair<uintE, uintE>> found_possible_edges =
                    sequence<std::pair<uintE, uintE>>(representative_starts.size());
            sequence<bool> is_edge = sequence<bool>(representative_starts.size());

            parallel_for(0, representative_starts.size(), [&](size_t i) {
                auto representative_node = representative_nodes[representative_starts[i]];
                auto id = representative_node->id.first;
                auto xor_sums = tree.get_tree_xor(id);
                bool is_real_edge = false;

                for (size_t ii = 0; ii < xor_sums.size(); ii++) {
                if (!is_real_edge) {
                    for (size_t ij = 0; ij < xor_sums[ii].size(); ij++) {
                        if (!is_real_edge) {
                            if (xor_sums[ii][ij].first != UINT_E_MAX
                                && xor_sums[ii][ij].second != UINT_E_MAX) {
                                auto u = xor_sums[ii][ij].first;
                                auto v = xor_sums[ii][ij].second;
                                if (u < n && v < n && existence_table.find(std::make_pair(u, v), false)) {
                                    auto node_a = tree.skip_list.find_representative(&tree.vertices[u]);
                                    auto node_b = tree.skip_list.find_representative(&tree.vertices[v]);

                                    if (node_a->id.first != node_b->id.first
                                            || node_a->id.second != node_b->id.second) {
                                        is_real_edge = true;
                                        found_possible_edges[i] = std::make_pair(u, v);
                                    }
                                }
                            }
                        }
                    }
                }
                }

                is_edge[i] = is_real_edge;
            });

            auto real_edges = parlay::pack(found_possible_edges, is_edge);

            std::pair<std::pair<uintE, uintE>, size_t> rep_empty =
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX),
                   0);

            auto rep_hash = [](const std::pair<uintE, uintE>& t) {
                    size_t l = std::min(std::get<0>(t), std::get<1>(t));
                    size_t r = std::max(std::get<0>(t), std::get<1>(t));
                    size_t key = (l << 32) + r;
                return parlay::hash64_2(key);
            };

            auto representative_to_real = gbbs::make_sparse_table<std::pair<uintE, uintE>, size_t>(real_edges.size(),
                rep_empty, rep_hash);

            auto potential_representative_edges = sequence<std::pair<uintE, uintE>>(real_edges.size(),
                    std::make_pair(UINT_E_MAX,
                        UINT_E_MAX));

            auto different_trees = sequence<bool>(real_edges.size(), false);

            parallel_for(0, real_edges.size(), [&](size_t i) {
                auto u = tree.vertices[real_edges[i].first];
                auto v = tree.vertices[real_edges[i].second];
                auto node_a = tree.skip_list.find_representative(&u);
                auto node_b = tree.skip_list.find_representative(&v);

                auto node_a_id = std::min(node_a->id.first, node_a->id.second);
                auto node_b_id = std::min(node_b->id.first, node_b->id.second);

                auto ru = std::min(node_a_id, node_b_id);
                auto rv = std::max(node_a_id, node_b_id);

                if (ru != rv) {
                    if (ru == rv)
                        std::cout << "ERROR: found edge in the same tree" << std::endl;
                    different_trees[i] = true;
                    potential_representative_edges[i] = std::make_pair((uintE)ru, (uintE)rv);
                }

                if (ru == UINT_E_MAX || rv == UINT_E_MAX)
                    std::cout << "ERROR: wrong indices found" << std::endl;
            });

            abort = false;

            auto representative_edges = parlay::pack(potential_representative_edges, different_trees);
            parallel_for(0, representative_edges.size(), [&](size_t i) {
                representative_to_real.insert_check(std::make_pair(representative_edges[i], i), &abort);
            });


            auto get_key = [&] (const std::pair<uintE, uintE> l) {
                return l.first;
            };
            parlay::integer_sort_inplace(make_slice(representative_edges), get_key);

            auto unique_bool_seq = sequence<bool>(representative_edges.size());
            parallel_for(0, representative_edges.size(), [&](size_t i) {
                unique_bool_seq[i] = ((i == 0) ||
                    ((representative_edges[i-1].first != representative_edges[i].first) ||
                    (representative_edges[i-1].second != representative_edges[i].second)));
            });
            auto unique_starts = parlay::pack_index(unique_bool_seq);
            auto unique_representative_edges = sequence<std::pair<uintE, uintE>>(unique_starts.size());
            auto unique_real_edges = sequence<std::pair<uintE, uintE>>(unique_starts.size());

            auto verts = sequence<uintE>(2 * unique_starts.size());

            parallel_for(0, unique_starts.size(), [&](size_t i){
                unique_representative_edges[i] = representative_edges[unique_starts[i]];
                auto real_index = representative_to_real.find(representative_edges[unique_starts[i]], UINT_E_MAX);

                if (real_index == UINT_E_MAX)
                    std::cout << "ERROR: real index for representative edge doesn't exist" << std::endl;

                auto unique_edge = real_edges[real_index];
                unique_real_edges[i] = unique_edge;

                verts[2 * i] = unique_representative_edges[i].first;
                verts[2 * i + 1] = unique_representative_edges[i].second;
            });

            KV_pairs empty =
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), std::make_pair(UINT_E_MAX, UINT_E_MAX));
            auto hash_pair = [](const std::pair<uintE, uintE>& t) {
                    size_t l = std::min(std::get<0>(t), std::get<1>(t));
                    size_t r = std::max(std::get<0>(t), std::get<1>(t));
                    size_t key = (l << 32) + r;
                return parlay::hash64_2(key);
            };

            std::pair<uintE, uintE> vert_empty = std::make_pair(UINT_E_MAX, UINT_E_MAX);
            auto vert_hash = [](const uintE& t){
                return parlay::hash64_2(t);
            };

            auto representative_edge_to_original =
                gbbs::make_sparse_table<K, V_pairs>(unique_representative_edges.size(), empty, hash_pair);

            parlay::sort_inplace(parlay::make_slice(verts));

            auto verts_bool_seq = sequence<bool>(verts.size());
            parallel_for(0, verts.size(), [&](size_t i) {
                verts_bool_seq[i] = ((i == 0) ||
                    (verts[i-1] != verts[i]));
            });
            auto remapped_verts = parlay::pack_index(verts_bool_seq);
            auto original_to_remapped_verts = gbbs::make_sparse_table<uintE, uintE>(remapped_verts.size(),
                vert_empty, vert_hash);

            parallel_for(0, remapped_verts.size(), [&] (size_t i){
                original_to_remapped_verts.insert_check(std::make_pair(verts[remapped_verts[i]], i), &abort);
            });

            auto remapped_edges = sequence<std::tuple<uintE, uintE, uintE>>(unique_representative_edges.size());

            parallel_for(0, unique_representative_edges.size(), [&](size_t i){
                auto edge = unique_representative_edges[i];
                auto vert1 = std::get<0>(edge);
                auto vert2 = std::get<1>(edge);
                auto u = std::min(vert1, vert2);
                auto v = std::max(vert1, vert2);

                auto remapped_u = original_to_remapped_verts.find(u, UINT_E_MAX);
                auto remapped_v = original_to_remapped_verts.find(v, UINT_E_MAX);

                if (remapped_u == UINT_E_MAX || remapped_v == UINT_E_MAX)
                    std::cout << "ERROR: remapping vertices" << std::endl;

                if (unique_real_edges[i].first == UINT_E_MAX || unique_real_edges[i].second == UINT_E_MAX)
                    std::cout << "ERROR: unique edges have UINT_E_MAX error" << std::endl;

                remapped_edges[i] = std::make_tuple(remapped_u, remapped_v, (uintE) 1);

                representative_edge_to_original.insert_check(std::make_pair(std::make_pair(std::min(remapped_u,
                        remapped_v), std::max(remapped_u, remapped_v)), unique_real_edges[i]), &abort);
            });

            auto new_graph = sym_graph_from_edges(remapped_edges, remapped_verts.size());

            auto spanning_forest = workefficient_sf::SpanningForest(new_graph);
            if(spanning_forest.size() == 0)
                non_empty_spanning_tree = false;
            else {
                auto original_edges = sequence<std::pair<uintE, uintE>>(spanning_forest.size(), std::make_pair(UINT_E_MAX, UINT_E_MAX));
                auto is_valid_edge = sequence<bool>(spanning_forest.size(), false);

                parallel_for(0, spanning_forest.size(), [&](size_t i) {
                    auto u = std::get<0>(spanning_forest[i]);
                    auto v = std::get<1>(spanning_forest[i]);

                    auto first = std::min(u, v);
                    auto second = std::max(u, v);

                    if ((u != 0) || (v != 0)) {
                        auto original_edge = representative_edge_to_original.find(std::make_pair(first, second),
                            std::make_pair(UINT_E_MAX, UINT_E_MAX));

                        if (original_edge.first == UINT_E_MAX || original_edge.second == UINT_E_MAX)
                            std::cout << "ERROR: representative edge not found in hashmap" << std::endl;

                        original_edges[i] = original_edge;

                        is_valid_edge[i] = true;
                    }
                });

                auto to_link_edges = parlay::pack(original_edges, is_valid_edge);

                // Original edges are guaranteed to be unique since representative edges are unique
                tree.batch_link(to_link_edges, edge_table);
            }
         }
    }

    bool is_connected(uintE u, uintE v) {
            auto uvert = &tree.vertices[u];
            auto vvert = &tree.vertices[v];
            return tree.skip_list.find_representative(uvert)
                == tree.skip_list.find_representative(vvert);
    }

    template <class Seq, class KY, class VL, class HH>
    void batch_deletion(const Seq& deletions, gbbs::sparse_table<KY, VL, HH>& edge_table,
            gbbs::sparse_table<KY, bool, HH>& existence_table) {
        auto non_empty_spanning_tree = true;
        auto first = true;
        sequence<std::pair<uintE, uintE>> edges_both_directions = sequence<std::pair<uintE, uintE>>(0);
        sequence<size_t> starts = sequence<size_t>(0);
        sequence<SkipList::SkipListElement*> representative_nodes = sequence<SkipList::SkipListElement*>(0);
        bool abort = false;

        while(non_empty_spanning_tree) {
            if (first) {
                edges_both_directions =  sequence<std::pair<uintE, uintE>>(2 * deletions.size());
                auto split_edges =
                    sequence<std::pair<uintE, uintE>>(deletions.size(),
                            std::make_pair(UINT_E_MAX, UINT_E_MAX));
                auto is_split_edge = sequence<bool>(deletions.size(), false);

                parallel_for(0, deletions.size(), [&](size_t i) {
                    auto [u, v] = deletions[i];
                    edges_both_directions[2 * i] = std::make_pair(u, v);
                    edges_both_directions[2 * i + 1] = std::make_pair(v, u);

                    auto edge_index = edge_table.find(std::make_pair(u, v), UINT_E_MAX);
                    if (edge_index != UINT_E_MAX && tree.edge_table[edge_index].twin != nullptr) {
                        is_split_edge[i] = true;
                        split_edges[i] = std::make_pair(
                            tree.edge_table[edge_index].id.first,
                            tree.edge_table[edge_index].id.second);
                    }
                });

                sequence<std::pair<uintE, uintE>>
                    edges_to_split = parlay::pack(split_edges, is_split_edge);

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

                parallel_for(0, starts.size() - 1, [&](size_t i) {
                    // update j's cutset data structure; need to be sequential because accessing same arrays
                    for (size_t j = starts[i]; j < starts[i+1]; j++) {
                        tree.add_edge_to_cutsets(edges_both_directions[j]);
                    }

                    SkipList::SkipListElement* our_vertex = &tree.vertices[edges_both_directions[starts[i]].first];
                    representative_nodes[i] =
                        tree.skip_list.find_representative(our_vertex);
                });

                tree.batch_cut(edges_to_split, edge_table);
            }

            first = false;

            if (!first) {
                parallel_for(0, starts.size() - 1, [&](size_t i) {
                    SkipList::SkipListElement* our_vertex = &tree.vertices[edges_both_directions[starts[i]].first];
                    representative_nodes[i] =
                        tree.skip_list.find_representative(our_vertex);
                });
            }

            parlay::sort_inplace(parlay::make_slice(representative_nodes));

            auto bool_seq = sequence<bool>(representative_nodes.size());
            parallel_for(0, representative_nodes.size(), [&](size_t i) {
                bool_seq[i] = (i == 0) ||
                       (edges_both_directions[i-1] != edges_both_directions[i]);
            });
            auto representative_starts = parlay::pack_index(bool_seq);

            sequence<std::pair<uintE, uintE>> found_possible_edges =
                    sequence<std::pair<uintE, uintE>>(representative_starts.size());
            sequence<bool> is_edge = sequence<bool>(representative_starts.size());

            parallel_for(0, representative_starts.size(), [&](size_t i) {
                auto representative_node = representative_nodes[representative_starts[i]];
                auto id = representative_node->id.first;
                auto xor_sums = tree.get_tree_xor(id);
                bool is_real_edge = false;

                for (size_t ii = 0; ii < xor_sums.size(); ii++) {
                if (!is_real_edge) {
                    for (size_t ij = 0; ij < xor_sums[ii].size(); ij++) {
                        if (!is_real_edge) {
                            if (xor_sums[ii][ij].first != UINT_E_MAX
                                && xor_sums[ii][ij].second != UINT_E_MAX) {
                                auto u = xor_sums[ii][ij].first;
                                auto v = xor_sums[ii][ij].second;
                                if (u < n && v < n && existence_table.find(std::make_pair(u, v), false)) {
                                    auto node_a = tree.skip_list.find_representative(&tree.vertices[u]);
                                    auto node_b = tree.skip_list.find_representative(&tree.vertices[v]);

                                    if (node_a->id.first != node_b->id.first
                                            || node_a->id.second != node_b->id.second) {
                                        is_real_edge = true;
                                        found_possible_edges[i] = std::make_pair(u, v);
                                    }
                                }
                            }
                        }
                    }
                }
                }

                is_edge[i] = is_real_edge;
            });

            auto real_edges = parlay::pack(found_possible_edges, is_edge);

            std::pair<std::pair<uintE, uintE>, size_t> rep_empty =
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX),
                   0);

            auto rep_hash = [](const std::pair<uintE, uintE>& t) {
                    size_t l = std::min(std::get<0>(t), std::get<1>(t));
                    size_t r = std::max(std::get<0>(t), std::get<1>(t));
                    size_t key = (l << 32) + r;
                return parlay::hash64_2(key);
            };

            auto representative_to_real = gbbs::make_sparse_table<std::pair<uintE, uintE>, size_t>(real_edges.size(),
                rep_empty, rep_hash);

            auto potential_representative_edges = sequence<std::pair<uintE, uintE>>(real_edges.size(),
                    std::make_pair(UINT_E_MAX,
                        UINT_E_MAX));

            auto different_trees = sequence<bool>(real_edges.size(), false);

            parallel_for(0, real_edges.size(), [&](size_t i) {
                auto u = tree.vertices[real_edges[i].first];
                auto v = tree.vertices[real_edges[i].second];
                auto node_a = tree.skip_list.find_representative(&u);
                auto node_b = tree.skip_list.find_representative(&v);

                auto node_a_id = std::min(node_a->id.first, node_a->id.second);
                auto node_b_id = std::min(node_b->id.first, node_b->id.second);

                auto ru = std::min(node_a_id, node_b_id);
                auto rv = std::max(node_a_id, node_b_id);

                if (ru != rv) {
                    if (ru == rv)
                        std::cout << "ERROR: found edge in the same tree" << std::endl;
                    different_trees[i] = true;
                    potential_representative_edges[i] = std::make_pair((uintE)ru, (uintE)rv);
                }

                if (ru == UINT_E_MAX || rv == UINT_E_MAX)
                    std::cout << "ERROR: wrong indices found" << std::endl;
            });

            abort = false;

            auto representative_edges = parlay::pack(potential_representative_edges, different_trees);
            parallel_for(0, representative_edges.size(), [&](size_t i) {
                representative_to_real.insert_check(std::make_pair(representative_edges[i], i), &abort);
            });


            auto get_key = [&] (const std::pair<uintE, uintE> l) {
                return l.first;
            };
            parlay::integer_sort_inplace(make_slice(representative_edges), get_key);

            auto unique_bool_seq = sequence<bool>(representative_edges.size());
            parallel_for(0, representative_edges.size(), [&](size_t i) {
                unique_bool_seq[i] = ((i == 0) ||
                    ((representative_edges[i-1].first != representative_edges[i].first) ||
                    (representative_edges[i-1].second != representative_edges[i].second)));
            });
            auto unique_starts = parlay::pack_index(unique_bool_seq);
            auto unique_representative_edges = sequence<std::pair<uintE, uintE>>(unique_starts.size());
            auto unique_real_edges = sequence<std::pair<uintE, uintE>>(unique_starts.size());

            auto verts = sequence<uintE>(2 * unique_starts.size());

            parallel_for(0, unique_starts.size(), [&](size_t i){
                unique_representative_edges[i] = representative_edges[unique_starts[i]];
                auto real_index = representative_to_real.find(representative_edges[unique_starts[i]], UINT_E_MAX);

                if (real_index == UINT_E_MAX)
                    std::cout << "ERROR: real index for representative edge doesn't exist" << std::endl;

                auto unique_edge = real_edges[real_index];
                unique_real_edges[i] = unique_edge;

                verts[2 * i] = unique_representative_edges[i].first;
                verts[2 * i + 1] = unique_representative_edges[i].second;
            });

            KV_pairs empty =
                std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), std::make_pair(UINT_E_MAX, UINT_E_MAX));
            auto hash_pair = [](const std::pair<uintE, uintE>& t) {
                    size_t l = std::min(std::get<0>(t), std::get<1>(t));
                    size_t r = std::max(std::get<0>(t), std::get<1>(t));
                    size_t key = (l << 32) + r;
                return parlay::hash64_2(key);
            };

            std::pair<uintE, uintE> vert_empty = std::make_pair(UINT_E_MAX, UINT_E_MAX);
            auto vert_hash = [](const uintE& t){
                return parlay::hash64_2(t);
            };

            auto representative_edge_to_original =
                gbbs::make_sparse_table<K, V_pairs>(unique_representative_edges.size(), empty, hash_pair);

            parlay::sort_inplace(parlay::make_slice(verts));

            auto verts_bool_seq = sequence<bool>(verts.size());
            parallel_for(0, verts.size(), [&](size_t i) {
                verts_bool_seq[i] = ((i == 0) ||
                    (verts[i-1] != verts[i]));
            });
            auto remapped_verts = parlay::pack_index(verts_bool_seq);
            auto original_to_remapped_verts = gbbs::make_sparse_table<uintE, uintE>(remapped_verts.size(),
                vert_empty, vert_hash);

            parallel_for(0, remapped_verts.size(), [&] (size_t i){
                original_to_remapped_verts.insert_check(std::make_pair(verts[remapped_verts[i]], i), &abort);
            });

            auto remapped_edges = sequence<std::tuple<uintE, uintE, uintE>>(unique_representative_edges.size());

            parallel_for(0, unique_representative_edges.size(), [&](size_t i){
                auto edge = unique_representative_edges[i];
                auto vert1 = std::get<0>(edge);
                auto vert2 = std::get<1>(edge);
                auto u = std::min(vert1, vert2);
                auto v = std::max(vert1, vert2);

                auto remapped_u = original_to_remapped_verts.find(u, UINT_E_MAX);
                auto remapped_v = original_to_remapped_verts.find(v, UINT_E_MAX);

                if (remapped_u == UINT_E_MAX || remapped_v == UINT_E_MAX)
                    std::cout << "ERROR: remapping vertices" << std::endl;

                if (unique_real_edges[i].first == UINT_E_MAX || unique_real_edges[i].second == UINT_E_MAX)
                    std::cout << "ERROR: unique edges have UINT_E_MAX error" << std::endl;

                remapped_edges[i] = std::make_tuple(remapped_u, remapped_v, (uintE) 1);

                representative_edge_to_original.insert_check(std::make_pair(std::make_pair(std::min(remapped_u,
                        remapped_v), std::max(remapped_u, remapped_v)), unique_real_edges[i]), &abort);
            });

            auto new_graph = sym_graph_from_edges(remapped_edges, remapped_verts.size());

            auto spanning_forest = workefficient_sf::SpanningForest(new_graph);
            if(spanning_forest.size() == 0)
                non_empty_spanning_tree = false;
            else {
                auto original_edges = sequence<std::pair<uintE, uintE>>(spanning_forest.size(), std::make_pair(UINT_E_MAX, UINT_E_MAX));
                auto is_valid_edge = sequence<bool>(spanning_forest.size(), false);

                parallel_for(0, spanning_forest.size(), [&](size_t i) {
                    auto u = std::get<0>(spanning_forest[i]);
                    auto v = std::get<1>(spanning_forest[i]);

                    auto first = std::min(u, v);
                    auto second = std::max(u, v);

                    if ((u != 0) || (v != 0)) {
                        auto original_edge = representative_edge_to_original.find(std::make_pair(first, second),
                            std::make_pair(UINT_E_MAX, UINT_E_MAX));

                        if (original_edge.first == UINT_E_MAX || original_edge.second == UINT_E_MAX)
                            std::cout << "ERROR: representative edge not found in hashmap" << std::endl;

                        original_edges[i] = original_edge;

                        is_valid_edge[i] = true;
                    }
                });

                auto to_link_edges = parlay::pack(original_edges, is_valid_edge);

                // Original edges are guaranteed to be unique since representative edges are unique
                tree.batch_link(to_link_edges, edge_table);
            }
         }
    }
};

void RunConnectivityTest() {
        std::cout << "Connectivity Test" << std::endl;
}

template <class W>
inline void RunConnectivity(BatchDynamicEdges<W>& batch_edge_list, long batch_size, bool compare_exact,
        size_t offset, size_t n, int copies, size_t m, double pb) {

        auto batch = batch_edge_list.edges;
        auto cutset = Connectivity(n, copies, batch.size(), pb);

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

                cutset.batch_insertion(batch_insertions, edge_table, existence_table);

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

                cutset.batch_deletion(batch_deletions, edge_table, existence_table);
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

            cutset.batch_insertion(batch_insertions, edge_table, existence_table);

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

            cutset.batch_deletion(batch_deletions, edge_table, existence_table);
            auto runtime = t.stop();
            std::cout << "runtime: " << runtime << std::endl;

            sequence<int> correct = sequence<int>(batch_insertions.size(), false);
            parallel_for(0, batch_insertions.size(), [&](size_t i) {
                correct[i] = cutset.is_connected(batch_insertions[i].first, batch_insertions[i].second);
            });
            auto num_correct = parlay::scan_inplace(correct);

            std::cout << "fraction correct: " << (num_correct * 1.0)/batch_insertions.size() << std::endl;
    }
}

}  // namespace gbbs
