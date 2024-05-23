#include <unordered_set>
#include <stack>

#include "gbbs/gbbs.h"
#include "gbbs/dynamic_graph_io.h"
#include "gbbs/helpers/sparse_table.h"
#include "benchmarks/KCore/JulienneDBS17/KCore.h"
#include "benchmarks/BatchDynamicConnectivity/SkipList/SkipList.h"

namespace gbbs {

sequence<sequence<std::pair<uintE, uintE>>> make_values(uintE a, uintE b, double pb, int copies, size_t m) {
        return sequence<sequence<std::pair<uintE, uintE>>>(copies, sequence<std::pair<uintE, uintE>>(
                    ceil(log(m)/log(pb)), std::make_pair(a, b)));
}

struct ETTree {
   sequence<SkipList::SkipListElement> edge_table;
   SkipList skip_list;
   sequence<SkipList::SkipListElement> vertices;
   int copies;
   size_t m;
   parlay::random cutset_rng = parlay::random(time(0));
   double pb;

   ETTree() {}

   ETTree(size_t n, int copies_, size_t m_, double pb_) {
        skip_list = SkipList(n);
        edge_table = sequence<SkipList::SkipListElement>(2 * m_);
        pb = pb_;
        m = m_;
        copies = copies_;

        vertices = sequence<SkipList::SkipListElement>(n);
        auto joins = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(n);
        parallel_for(0, n, [&] (size_t i) {
            vertices[i] = skip_list.create_node(i, nullptr, nullptr, make_values(0, 0, pb, copies, m), nullptr, true,
                    std::make_pair(i, i), pb, copies, m, 1);
            joins[i] = std::make_pair(&vertices[i], &vertices[i]);
        });

        skip_list.batch_join(&joins);
    }

    void print_value(std::string label, SkipList::SkipListElement* v) {
            std::cout << label << ", id: " << v->id.first << ", " << v->id.second <<
            ", values: " << v->values[0][0][0].first << ", " << v->values[0][0][0].second << std::endl;
    }

    template <class KY, class VL, class HH>
    void link(uintE u, uintE v, gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
        auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);
        auto index_vu = edge_index_table.find(std::make_pair(v, u), UINT_E_MAX);

        if (index_uv == UINT_E_MAX || index_vu == UINT_E_MAX)
            std::cout << "THERE IS AN ERROR in edge_index_table" << std::endl;

        auto output = edge_table[index_uv];
        edge_table[index_uv] = skip_list.create_node(u, nullptr, nullptr, make_values(0, 0, pb, copies, m), nullptr, false,
                std::make_pair(u, v), pb, copies, m, 0);
        auto uv = &edge_table[index_uv];
        edge_table[index_vu] = skip_list.create_node(v, nullptr, nullptr, make_values(0, 0, pb, copies, m), uv, false,
                std::make_pair(v, u), pb, copies, m, 0);
        auto vu = &edge_table[index_vu];
        uv->twin = vu;
        vu->twin = uv;

        auto u_left = &vertices[u];
        auto v_left = &vertices[v];


        auto splits = sequence<SkipList::SkipListElement*>(2);
        splits[0] = u_left;
        splits[1] = v_left;
        auto results = skip_list.batch_split(&splits);

        auto u_right = results[0];
        auto v_right = results[1];

        auto joins = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(4);

        joins[0] = std::make_pair(u_left, uv);
        joins[1] = std::make_pair(uv, v_right);
        joins[2] = std::make_pair(v_left, vu);
        joins[3] = std::make_pair(vu, u_right);

        skip_list.batch_join(&joins);
    }

    template <class KY, class VL, class HH>
    void cut(int u, int v, gbbs::sparse_table<KY, VL, HH>& edge_index_table){
            auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);
            if (index_uv == UINT_E_MAX)
                std::cout << "There is an error in edge_index_table" << std::endl;

            auto uv = &edge_table[index_uv];
            auto vu = uv->twin;

            auto u_left = uv->get_left(0);
            auto v_left = vu->get_left(0);

            auto splits = sequence<SkipList::SkipListElement*>(2);
            splits[0] = uv;
            splits[1] = vu;
            auto results = skip_list.batch_split(&splits);

            auto v_right = results[0];
            auto u_right = results[1];

            splits = sequence<SkipList::SkipListElement*>(2);
            splits[0] = u_left;
            splits[1] = v_left;
            results = skip_list.batch_split(&splits);

            uv->twin = nullptr;
            vu->twin = nullptr;

            auto joins = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(2);

            joins[0] = std::make_pair(u_left, u_right);
            joins[1] = std::make_pair(v_left, v_right);

            skip_list.batch_join(&joins);

            edge_table[index_uv] = SkipList::SkipListElement();
            auto index_vu = edge_index_table.find(std::make_pair(v, u), UINT_E_MAX);
            edge_table[index_vu] = SkipList::SkipListElement();
    }

    template <class KY, class VL, class HH>
    void batch_link_sequential(sequence<std::pair<uintE, uintE>>& links,
            gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
            std::cout << "started link sequential" << std::endl;
            for(size_t i = 0; i < links.size(); i++) {
                link(links[i].first, links[i].second, edge_index_table);
            }
            std::cout << "ended link sequential" << std::endl;
    }

    template <class KY, class VL, class HH>
    void batch_link(sequence<std::pair<uintE, uintE>>& links, gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
        if (links.size() <= 75) {
                batch_link_sequential(links, edge_index_table);
                return;
        }

        std::cout << "started batch links" << std::endl;

        sequence<std::pair<uintE, uintE>> links_both_dirs = sequence<std::pair<uintE, uintE>>(2 * links.size());
        parallel_for(0, links.size(), [&] (size_t i) {
                links_both_dirs[2 * i] = links[i];
                links_both_dirs[2 * i + 1] = std::make_pair(links[i].second, links[i].first);
        });

        std::cout << "ended links both directions" << std::endl;

        auto get_key = [&] (const std::pair<uintE, uintE>& elm) { return elm.first; };
        parlay::integer_sort_inplace(parlay::make_slice(links_both_dirs), get_key);

        auto split_successors = sequence<SkipList::SkipListElement*>(2 * links.size());
        auto splits = sequence<SkipList::SkipListElement*>(2 * links.size(), nullptr);

        std::cout << "splits created" << std::endl;

        parallel_for(0, 2 * links.size(), [&] (size_t i) {
            uintE u, v;
            u = links_both_dirs[i].first;
            v = links_both_dirs[i].second;

            if (i == 2 * links.size() - 1 || u != links_both_dirs[i+1].first) {
                    splits[i] = &vertices[u];
            }

            if (u < v) {
                    auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);
                    if (index_uv == UINT_E_MAX)
                        std::cout << "This is an error in edge_index_table in recursive insertions" << std::endl;

                    edge_table[index_uv] = skip_list.create_node(u, nullptr, nullptr, make_values(0, 0, pb, copies, m),
                            nullptr, false, std::make_pair(u, v), pb, copies, m, 0);
                    auto uv = &edge_table[index_uv];

                    auto index_vu = edge_index_table.find(std::make_pair(v, u), UINT_E_MAX);
                    if (index_vu == UINT_E_MAX)
                        std::cout << "This is an error in edge_index_table in recursive insertions" << std::endl;

                    edge_table[index_vu] = skip_list.create_node(v, nullptr, nullptr, make_values(0, 0, pb, copies, m),
                            uv, false, std::make_pair(v, u), pb, copies, m, 0);
                    auto vu = &edge_table[index_vu];
                    vu->twin = uv;
                    uv->twin = vu;
            }
        });

        std::cout << "splits ended" << std::endl;

        auto bool_seq = parlay::delayed_seq<bool>(splits.size(), [&] (size_t i) {
                return (splits[i] != nullptr);
        });

        auto element_indices = parlay::pack_index(bool_seq);
        auto filtered_splits = sequence<SkipList::SkipListElement*>(element_indices.size());
        parallel_for(0, element_indices.size(), [&] (size_t i) {
            filtered_splits[i] = splits[element_indices[i]];
        });

        auto results = skip_list.batch_split(&filtered_splits);

        parallel_for(0, element_indices.size(), [&] (size_t i) {
            auto split_index = element_indices[i];
            split_successors[split_index] = results[i];
        });

        std::cout << "filtered splits ended" << std::endl;

        // If x has new neighbors y_1, y_2, ..., y_k, join (x, x) to (x, y_1). Join
        // (y_i,x) to (x, y_{i+1}) for each i < k. Join (y_k, x) to succ(x)

        auto joins = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(2 * links.size(),
                std::make_pair(nullptr, nullptr));

        std::cout << "starting to make joins" << std::endl;
        parallel_for(0, links.size(), [&] (size_t i) {
            uintE u, v;
            u = links_both_dirs[i].first;
            v = links_both_dirs[i].second;

            auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);

            std::cout << "looking in index table" << std::endl;
            if (index_uv == UINT_E_MAX)
                std::cout << "This is an error in edge_index_table in recursive insertions" << std::endl;

            SkipList::SkipListElement* uv = &edge_table[index_uv];
            SkipList::SkipListElement* vu = uv -> twin;

            if (i == 0 || u != links_both_dirs[i-1].first) {
                joins[2*i] = std::make_pair(&vertices[u], uv);
            }

            if (i == 2 * links.size() - 1 || u != links_both_dirs[i+1].first) {
                joins[2*i + 1] = std::make_pair(vu, split_successors[i]);
                if (split_successors[i] == nullptr)
                    std::cout << "ERROR: wrong successor" << std::endl;
            } else {
                uintE u2, v2;
                u2 = links_both_dirs[i+1].first;
                v2 = links_both_dirs[i+1].second;
                auto index_uv2 = edge_index_table.find(std::make_pair(u2, v2), UINT_E_MAX);
                if (index_uv2 == UINT_E_MAX)
                    std::cout << "This is an error in edge_index_table in recursive insertions uv2" << std::endl;

                auto found_element = &edge_table[index_uv2];
                joins[2*i + 1] = std::make_pair(vu, found_element);
            }
        });

        std::cout << "joins created" << std::endl;

        sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>> filtered =
            parlay::filter(joins, [&] (const std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>& e) {

                if ((e.first == nullptr && e.second != nullptr) || (e.first != nullptr && e.second != nullptr))
                    std::cout << "ERROR: join error nullptr" << std::endl;
                return e.first != nullptr && e.second != nullptr;
            });

        std::cout << "batch joins created" << std::endl;
        skip_list.batch_join(&filtered);
        std::cout << "batch joins ended" << std::endl;
    }

    template <class KY, class VL, class HH>
    void batch_cut_sequential(sequence<std::pair<uintE, uintE>>& cuts,
            gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
            for (size_t i = 0; i < cuts.size(); i++)
                cut(cuts[i].first, cuts[i].second, edge_index_table);
    }

    template <class KY, class VL, class HH>
    void batch_cut_recurse(sequence<std::pair<uintE, uintE>>& cuts,
        gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
            sequence<SkipList::SkipListElement*> join_targets =
                sequence<SkipList::SkipListElement*>(4 * cuts.size(), nullptr);
            sequence<SkipList::SkipListElement*> edge_elements =
                sequence<SkipList::SkipListElement*>(cuts.size(), nullptr);

            parlay::random rng = parlay::random(time(0));

            if (cuts.size() <= 75) {
                    batch_cut_sequential(cuts, edge_index_table);
                    return;
            }
            sequence<bool> ignored = sequence<bool>(cuts.size(), true);

            parallel_for(0, cuts.size(), [&] (size_t i) {
                rng.fork(i);
                rng = rng.next();
                bool rand_val = rng.rand() % 100 == 0;

                ignored[i] = rand_val;
                if (!ignored[i]) {
                    uintE u, v;
                    u = cuts[i].first;
                    v = cuts[i].second;

                    auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);
                    if (index_uv == UINT_E_MAX)
                        std::cout << "This is an error in edge_index_table in batch deletions" << std::endl;

                    SkipList::SkipListElement* uv = &edge_table[index_uv];
                    SkipList::SkipListElement* vu = uv->twin;

                    edge_elements[i] = uv;
                    uv -> split_mark = true;
                    vu -> split_mark = true;
                }

                rng = rng.next();

            });

            parallel_for(0, cuts.size(), [&] (size_t i) {
                    if (!ignored[i]) {
                        SkipList::SkipListElement* uv = edge_elements[i];
                        SkipList::SkipListElement* vu = uv->twin;

                        SkipList::SkipListElement* left_target = uv->get_left(0);
                        if (left_target -> split_mark) {
                            join_targets[4 * i] = nullptr;
                        } else {
                            SkipList::SkipListElement* right_target = vu->get_right(0);
                            while (right_target->split_mark) {
                                right_target = right_target->twin->get_right(0);
                            }
                            join_targets[4 * i] = left_target;
                            join_targets[4 * i + 1] = right_target;
                        }
                        left_target = vu->get_left(0);
                        if(left_target -> split_mark) {
                            join_targets[4 * i + 2] = nullptr;
                        } else {
                            SkipList::SkipListElement* right_target = uv -> get_right(0);
                            while (right_target -> split_mark) {
                                    right_target = right_target->twin->get_right(0);
                            }
                            join_targets[4 * i + 2] = left_target;
                            join_targets[4 * i + 3] = right_target;
                        }
                    }
            });

            auto splits = sequence<SkipList::SkipListElement*>(4 * cuts.size());
            parallel_for(0, cuts.size(), [&] (size_t i) {
                    if (!ignored[i]) {
                        SkipList::SkipListElement* uv = edge_elements[i];
                        SkipList::SkipListElement* vu = uv->twin;

                        splits[4 * i] = uv;
                        splits[4 * i + 1] = vu;

                        SkipList::SkipListElement* predecessor = uv->get_left(0);
                        if (predecessor != nullptr) {
                            splits[4 * i + 2] = predecessor;
                        }

                        predecessor = vu->get_left(0);
                        if (predecessor != nullptr) {
                            splits[4 * i + 3] = predecessor;
                        }

                    }
           });

           sequence<SkipList::SkipListElement*> filtered =
                parlay::filter(splits, [&] (const SkipList::SkipListElement* e) {
                    return e != nullptr;
                });
           skip_list.batch_split(&filtered);

           auto joins = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(
                join_targets.size() / 2,
                std::make_pair(nullptr, nullptr));
           parallel_for(0, cuts.size(), [&] (size_t i) {
                    if (!ignored[i]) {
                        uintE u, v;
                        u = cuts[i].first;
                        v = cuts[i].second;

                        auto index_uv = edge_index_table.find(std::make_pair(u, v), UINT_E_MAX);
                        if (index_uv == UINT_E_MAX)
                            std::cout << "This is an error in edge_index_table in batch deletions" << std::endl;

                        edge_table[index_uv] = SkipList::SkipListElement();
                        auto index_vu = edge_index_table.find(std::make_pair(v, u), UINT_E_MAX);
                        if (index_vu == UINT_E_MAX)
                            std::cout << "This is an error in edge_index_table in batch deletions" << std::endl;

                        edge_table[index_vu] = SkipList::SkipListElement();

                        if (join_targets[4 * i] != nullptr) {
                            joins[2 * i] = std::make_pair(join_targets[4 * i], join_targets[4 * i + 1]);
                        }

                        if (join_targets[4 * i + 2] != nullptr) {
                            joins[2 * i + 1] = std::make_pair(join_targets[4 * i + 2], join_targets[4 * i + 3]);
                        }
                    }
            });
            sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>> filtered_joins =
                parlay::filter(joins, [&] (const std::pair<SkipList::SkipListElement*,
                            SkipList::SkipListElement*>& e) {
                    return e.first != nullptr || e.second != nullptr;
            });

            skip_list.batch_join(&filtered_joins);

            auto element_indices = parlay::pack_index(ignored);
            auto next_cuts_seq = sequence<std::pair<uintE, uintE>>(element_indices.size());
            parallel_for(0, next_cuts_seq.size(), [&] (size_t i){
                next_cuts_seq[i] = cuts[element_indices[i]];
            });
            batch_cut_recurse(next_cuts_seq, edge_index_table);
    }

    template <class KY, class VL, class HH>
    void batch_cut(sequence<std::pair<uintE, uintE>>& cuts, gbbs::sparse_table<KY, VL, HH>& edge_index_table) {
            if (cuts.size() <=  75) {
                    batch_cut_sequential(cuts, edge_index_table);
                    return;
            }

            batch_cut_recurse(cuts, edge_index_table);
    }

    // Dynamic connectivity specific method
    void add_edge_to_cutsets(std::pair<uintE, uintE> edge) {
        auto min_edge = std::min(edge.first, edge.second);
        auto max_edge = std::max(edge.first, edge.second);

        auto u = &vertices[edge.first];

        parallel_for(0, u->values[0].size(), [&](size_t ii) {
            parallel_for(0, u->values[0][ii].size(), [&](size_t ij) {
                    cutset_rng.fork(edge.first + ii * ij);
                    cutset_rng = cutset_rng.next();
                    auto base = uintE(floor(pow(pb, ij)));

                    if (base < 1)
                        base = 1;

                    auto rand_val_u = cutset_rng.rand() % base;
                    if (rand_val_u <= 0)
                        u->values[0][ii][ij] = std::make_pair(u->values[0][ii][ij].first ^ min_edge,
                                u->values[0][ii][ij].second ^ max_edge);

                    cutset_rng = cutset_rng.next();
            });
        });
    }

    sequence<sequence<std::pair<uintE, uintE>>> get_tree_xor(uintE v) {
        auto edge = &vertices[v];
        auto edge_sum = skip_list.get_xor(edge);
        return edge_sum;
    }

    size_t get_tree_size(uintE v) {
        auto edge = &vertices[v];
        auto edge_sum = skip_list.get_sum(edge);
        return edge_sum;
    }

    bool is_connected(uintE u, uintE v) {
            auto uu = vertices[u];
            auto vv = vertices[v];

            return skip_list.find_representative(&uu) == skip_list.find_representative(&vv);
    }
};

void RunETTree(double pb, int copies, size_t m) {
        using K = std::pair<uintE, uintE>;
        using V = uintE;
        using KV = std::pair<K, V>;

        KV empty =
            std::make_pair(std::make_pair(UINT_E_MAX, UINT_E_MAX), UINT_E_MAX);

        auto hash_pair = [](const std::pair<uintE, uintE>& t) {
            size_t l = std::min(std::get<0>(t), std::get<1>(t));
            size_t r = std::max(std::get<0>(t), std::get<1>(t));
            size_t key = (l << 32) + r;
            return parlay::hash64_2(key);
        };

        auto edge_index_table =
            gbbs::make_sparse_table<K, V>(2 * 8, empty, hash_pair);
        bool abort = false;

        auto tree = ETTree((size_t) 10, copies, m, pb);

        sequence<std::pair<uintE, uintE>> links = sequence<std::pair<uintE, uintE>>(8);
        links[0] = std::make_pair(2, 3);
        edge_index_table.insert_check(std::make_pair(std::make_pair(2, 3), 0), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(3, 2), 1), &abort);
        links[1] = std::make_pair(3, 4);
        edge_index_table.insert_check(std::make_pair(std::make_pair(3, 4), 2), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(4, 3), 3), &abort);
        links[2] = std::make_pair(0, 1);
        edge_index_table.insert_check(std::make_pair(std::make_pair(0, 1), 4), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(1, 0), 5), &abort);
        links[3] = std::make_pair(0, 8);
        edge_index_table.insert_check(std::make_pair(std::make_pair(0, 8), 6), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(8, 0), 7), &abort);
        links[4] = std::make_pair(7, 2);
        edge_index_table.insert_check(std::make_pair(std::make_pair(7, 2), 8), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(2, 7), 9), &abort);
        links[5] = std::make_pair(1, 2);
        edge_index_table.insert_check(std::make_pair(std::make_pair(1, 2), 10), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(2, 1), 11), &abort);
        links[6] = std::make_pair(6, 5);
        edge_index_table.insert_check(std::make_pair(std::make_pair(6, 5), 12), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(5, 6), 13), &abort);
        links[7] = std::make_pair(6, 9);
        edge_index_table.insert_check(std::make_pair(std::make_pair(6, 9), 14), &abort);
        edge_index_table.insert_check(std::make_pair(std::make_pair(9, 6), 15), &abort);

        tree.batch_link(links, edge_index_table);
        std::cout << "Connected 2, 3: " << tree.is_connected(2, 3) << std::endl;
        std::cout << "Connected 2, 0: " << tree.is_connected(2, 0) << std::endl;
        std::cout << "Connected 2, 4: " << tree.is_connected(2, 4) << std::endl;
        std::cout << "Connected 3, 7: " << tree.is_connected(3, 7) << std::endl;
        std::cout << "Connected 3, 8: " << tree.is_connected(3, 8) << std::endl;
        std::cout << "Connected 1, 8: " << tree.is_connected(1, 8) << std::endl;
        std::cout << "Connected 1, 5: " << tree.is_connected(1, 5) << std::endl;
        std::cout << "Connected 5, 9: " << tree.is_connected(5, 9) << std::endl;
        std::cout << "Connected 3, 6: " << tree.is_connected(3, 6) << std::endl;

        std::cout << "Getting subsequence sums: " << std::endl;
        auto a = tree.get_tree_size(2);
        auto b = tree.get_tree_size(5);
        auto c = tree.get_tree_xor(2);
        auto e = tree.get_tree_size(3);
        auto f = tree.get_tree_size(9);
        auto d = tree.get_tree_xor(5);

        std::cout << "v: 2, size: " << a << ", " << "v: 5, size: " << b << ", " << "v: 3, size: " <<
            e << ", " << "v: 9, size: " << f << std::endl;
        std::cout << "v: 2, parent: 2: " << c[0][0].first << ", " << c[0][0].second << std::endl;
        std::cout << "v: 5, parent: 5: " << d[0][0].first << ", " << d[0][0].second << std::endl;

        tree.add_edge_to_cutsets(std::make_pair(1, 6));
        tree.add_edge_to_cutsets(std::make_pair(6, 1));
        tree.add_edge_to_cutsets(std::make_pair(1, 9));

        std::cout << "v: 1, value: " << tree.vertices[1].values[0][0][0].first << ", "
            << tree.vertices[1].values[0][0][0].second
            << std::endl;
        std::cout << "v: 6, value: " << tree.vertices[6].values[0][0][0].first << ", "
            << tree.vertices[6].values[0][0][0].second
            << std::endl;
        std::cout << "v: 9, value: " << tree.vertices[9].values[0][0][0].first << ", "
            << tree.vertices[9].values[0][0][0].second
            << std::endl;

        sequence<std::pair<uintE, uintE>> cuts = sequence<std::pair<uintE, uintE>>(4);
        cuts[0] = std::make_pair(0, 8);
        cuts[1] = std::make_pair(2, 3);
        cuts[2] = std::make_pair(6, 9);
        cuts[3] = std::make_pair(0, 1);

        tree.batch_cut(cuts, edge_index_table);
        std::cout << "After Batch Cut" << std::endl;
        std::cout << "Connected 2, 3: " << tree.is_connected(2, 3) << std::endl;
        std::cout << "Connected 2, 0: " << tree.is_connected(2, 0) << std::endl;
        std::cout << "Connected 2, 4: " << tree.is_connected(2, 4) << std::endl;
        std::cout << "Connected 3, 7: " << tree.is_connected(3, 7) << std::endl;
        std::cout << "Connected 3, 4: " << tree.is_connected(3, 4) << std::endl;
        std::cout << "Connected 1, 8: " << tree.is_connected(1, 8) << std::endl;
        std::cout << "Connected 1, 5: " << tree.is_connected(1, 5) << std::endl;
        std::cout << "Connected 5, 9: " << tree.is_connected(5, 9) << std::endl;
        std::cout << "Connected 5, 6: " << tree.is_connected(5, 6) << std::endl;
}

}  // namespace gbbs
