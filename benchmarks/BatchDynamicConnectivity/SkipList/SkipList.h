#include <unordered_set>
#include <stack>

#include "gbbs/gbbs.h"
#include "gbbs/dynamic_graph_io.h"
#include "gbbs/helpers/sparse_table.h"
#include "benchmarks/KCore/JulienneDBS17/KCore.h"


namespace gbbs {

struct SkipList {

    struct SkipListElement {
        size_t height;
        size_t lowest_needs_update = 0;

        using pointers = std::pair<SkipListElement*, SkipListElement*>;

        using height_array = sequence<pointers>;
        using values_array = sequence<sequence<sequence<std::pair<uintE, uintE>>>>;

        height_array neighbors;
        values_array values;

        sequence<size_t> size;

        uintE update_level_xor;
        uintE update_level_sum;
        bool split_mark;
        SkipListElement* twin;
        bool is_vertex;
        std::pair<uintE, uintE> id;
        double probability_base;
        int num_duplicates;

        SkipListElement(): height(0), lowest_needs_update(0), size(1) {
            update_level_xor = UINT_E_MAX;
            update_level_sum = UINT_E_MAX;
            split_mark = false;
            twin = nullptr;
            is_vertex = false;
            id = std::make_pair(UINT_E_MAX, UINT_E_MAX);
            probability_base = 2;
            num_duplicates = 2;
        }

        SkipListElement(size_t _h, SkipListElement* _r, SkipListElement* _l,
                sequence<sequence<std::pair<uintE, uintE>>> _vals,
                SkipListElement* twin_ = nullptr, bool is_vertex_ = false, std::pair<uintE, uintE> id_ =
                std::make_pair(UINT_E_MAX, UINT_E_MAX), double pb = 2, int num_dup = 2, size_t m = 10, uintE size_ = 1):
            height(_h), lowest_needs_update(_h) {
                neighbors.resize(_h);
                update_level_xor = UINT_E_MAX;
                update_level_sum = UINT_E_MAX;
                neighbors[0].first = _l;
                neighbors[0].second = _r;
                values.resize(_h);

                size.resize(_h);

                values[0] = _vals;
                split_mark = false;
                twin = twin_;
                is_vertex = is_vertex_;
                id = id_;
                probability_base = pb;
                num_duplicates = num_dup;

                size[0] = size_;
                parallel_for(1, _h, [&](size_t i){
                    values[i] = sequence<sequence<std::pair<uintE, uintE>>>(num_duplicates,
                            sequence<std::pair<uintE, uintE>>(ceil(log(m)/log(pb)), std::make_pair(0, 0)));

                    size[i] = size_;
                });
        }

        inline void set_left_pointer(size_t height, SkipListElement* left) {
            neighbors[height].first = left;
        }

        inline void set_right_pointer(size_t height, SkipListElement* right) {
            neighbors[height].second = right;
        }

        inline bool CASleft(size_t level,
                SkipListElement* old_left, SkipListElement* new_left) {
               return gbbs::atomic_compare_and_swap(&neighbors[level].first, old_left, new_left);
        }

        inline bool CASright(size_t level,
                SkipListElement* old_right, SkipListElement* new_right) {
               return gbbs::atomic_compare_and_swap(&neighbors[level].second, old_right, new_right);
        }

        inline SkipListElement* get_left(size_t height) {
                return neighbors[height].first;
        }

        inline SkipListElement* get_right(size_t height) {
                return neighbors[height].second;
        }
    };

    size_t n;
    parlay::random rng = parlay::random(time(0));

    SkipList(): n(0) {}

    SkipList(size_t _n): n(_n) {}

    SkipListElement create_node(size_t index, SkipListElement* left, SkipListElement* right,
            sequence<sequence<std::pair<uintE, uintE>>> vals,
            SkipListElement* twin = nullptr, bool is_vertex = false,
            std::pair<uintE, uintE> id = std::make_pair(UINT_E_MAX, UINT_E_MAX),
            double pb = 2, int num_dup = 2, size_t m = 10, uintE value = 1) {
        rng.fork(index);
        rng = rng.next();
        auto rand_val = rng.rand() % UINT_E_MAX;
        rng = rng.next();

        size_t cur_height = 1;
        while (rand_val & 1) {
                rand_val >>= 1;
                cur_height++;
        }

        auto height = std::min(cur_height, (size_t) 32);

        auto node = SkipListElement(height, left, right, vals, twin, is_vertex, id, pb, num_dup, m, value);
        return node;
    }

    SkipListElement* find_left_parent(size_t level, SkipListElement* this_element) {
            SkipListElement* cur_element = this_element;
            SkipListElement* start_element = this_element;
            if (cur_element->height > level + 1)
                return cur_element;

            cur_element = cur_element->neighbors[level].first;

            while (cur_element != nullptr && cur_element != start_element) {
                if (cur_element->height > level + 1)
                    return cur_element;
                cur_element = cur_element->neighbors[level].first;
            }
            return nullptr;
    }

    SkipListElement* find_right_parent(size_t level, SkipListElement* this_element) {
            SkipListElement* cur_element = this_element;
            SkipListElement* start_element = this_element;

            if (cur_element->height > level+1)
                return cur_element;
            cur_element = cur_element->neighbors[level].second;

            while (cur_element != nullptr && cur_element != start_element) {
                if (cur_element->height > level + 1)
                    return cur_element;
                cur_element = cur_element->neighbors[level].second;
            }
            return nullptr;
    }

    SkipListElement* find_representative(SkipListElement* this_element) {
        SkipListElement* cur_element = this_element;
        SkipListElement* seen_element = nullptr;

        size_t current_level = cur_element->height - 1;

        while(cur_element->neighbors[current_level].second != nullptr &&
                seen_element != cur_element) {
                if (seen_element == nullptr || cur_element < seen_element)
                    seen_element = cur_element;
                cur_element = cur_element->neighbors[current_level].second;
                auto top_level = cur_element->height-1;
                if (current_level < top_level) {
                        current_level = top_level;
                        seen_element = nullptr;
                }
        }

        if (seen_element == cur_element) {
                return seen_element;
        } else {
                // skiplist is not cyclic so find the leftmost element
                while (cur_element->neighbors[current_level].first != nullptr) {
                        cur_element = cur_element->neighbors[current_level].first;
                        current_level = cur_element->height - 1;
                }
                return cur_element;
        }
    }

    void join(SkipListElement* left, SkipListElement* right) {
            size_t level = 0;

            while(left != nullptr && right != nullptr) {
                    if (left->neighbors[level].second == nullptr &&
                                left->CASright(level, nullptr, right)) {
                            right->CASleft(level, nullptr, left);
                            left = find_left_parent(level, left);
                            right = find_right_parent(level, right);
                            level++;
                    } else {
                            return;
                    }
            }
    }

    SkipListElement* split(SkipListElement* this_element) {
            SkipListElement* successor = nullptr;
            SkipListElement* cur_element = this_element;

            size_t level = 0;

            while(cur_element != nullptr) {
                SkipListElement* next = cur_element->neighbors[level].second;
                if (next != nullptr &&
                        cur_element->CASright(level, next, nullptr)) {
                        if (level == 0) {
                           successor = next;
                        }

                        next->CASleft(level, cur_element, nullptr);
                        cur_element = find_left_parent(level, cur_element);
                        level++;
               }
               else {
                        break;
               }
            }
            return successor;
    }

    sequence<std::pair<SkipListElement*, SkipListElement*>>
        find_left_parents(size_t level, sequence<SkipListElement*>& elements) {

        sequence<std::pair<SkipListElement*, SkipListElement*>> parents = sequence<std::pair<SkipListElement*,
            SkipListElement*>>(elements.size());
        parallel_for(0, elements.size(), [&](size_t i) {
            parents[i] = std::make_pair(find_left_parent(level, elements[i]), elements[i]);
        });

        return parents;
    }

    /* The augmented skip list can take any arbitrary associative, commutative function. Here, it is
     * implemented using the XOR function. */
    void update_top_down_xor(size_t level, SkipListElement* this_element) {
            if (level == 0) {
                    if (this_element->height == 1) {
                        this_element->update_level_xor = UINT_E_MAX;
                    }
                    return;
            }

            if (this_element->update_level_xor < level) {
                    update_top_down_xor(level - 1, this_element);
            }

            sequence<sequence<std::pair<uintE, uintE>>> xor_total = this_element->values[level-1];
            SkipListElement* curr = this_element->neighbors[level-1].second;
            // only update if neighbor has height at most current level
            while (curr != nullptr && curr->height < level + 1) {
                    if (curr->update_level_xor != UINT_E_MAX && curr->update_level_xor < level) {
                            update_top_down_xor(level-1, curr);
                    }

                    parallel_for(0, xor_total.size(), [&] (size_t ii) {
                            parallel_for(0, xor_total[ii].size(), [&] (size_t ij) {
                                xor_total[ii][ij].first ^= curr->values[level-1][ii][ij].first;
                                xor_total[ii][ij].second ^= curr->values[level-1][ii][ij].second;
                            });
                    });
                    curr = curr->neighbors[level-1].second;
            }
            this_element->values[level] = xor_total;

            if(this_element->height == level+1) {
                    this_element->update_level_xor = UINT_E_MAX;
            }
    }


    /* The augmented skip list can take any arbitrary associative, commutative function. Here, it is
     * implemented using the SUM function. */
    void update_top_down_sum(size_t level, SkipListElement* this_element) {
            if (level == 0) {
                    if (this_element->height == 1) {
                        this_element->update_level_sum = UINT_E_MAX;
                    }
                    return;
            }

            if (this_element->update_level_sum < level) {
                    update_top_down_sum(level - 1, this_element);
            }

            size_t sum_total = this_element->size[level-1];
            SkipListElement* curr = this_element->neighbors[level-1].second;
            // only update if neighbor has height at most current level
            while (curr != nullptr && curr->height < level + 1) {
                    if (curr->update_level_sum != UINT_E_MAX && curr->update_level_sum < level) {
                            update_top_down_sum(level-1, curr);
                    }

                    sum_total += curr->size[level-1];
                    curr = curr->neighbors[level-1].second;
            }

            auto old_size = this_element->size[level];
            gbbs::atomic_compare_and_swap(&this_element->size[level], old_size, sum_total);

            if(this_element->height == level+1) {
                    this_element->update_level_sum = UINT_E_MAX;
            }
    }

    void batch_update_xor(sequence<std::pair<SkipListElement*,
            sequence<sequence<std::pair<uintE, uintE>>>>>* new_values) {
        auto top_nodes = sequence<SkipListElement*>(new_values->size(), nullptr);
        sequence<std::pair<SkipListElement*, sequence<sequence<std::pair<uintE, uintE>>>>>&
            new_values_ref = *new_values;
        if (new_values != nullptr) {
            parallel_for(0, new_values->size(), [&](size_t i) {
                    SkipListElement* this_element = new_values_ref[i].first;
                    sequence<sequence<std::pair<uintE, uintE>>>
                        this_element_values = new_values_ref[i].second;

                    parallel_for(0, this_element_values.size(), [&](size_t ii) {
                        parallel_for(0, this_element_values[ii].size(), [&](size_t ij) {
                            this_element->values[0][ii][ij] = this_element_values[ii][ij];
                        });
                    });

                    size_t level = 0;
                    SkipListElement* curr = this_element;
                    while(true) {
                        uintE curr_update_level = curr->update_level_xor;
                        if (curr_update_level == UINT_E_MAX && gbbs::atomic_compare_and_swap(&curr->update_level_xor,
                                UINT_E_MAX, (uintE) level)) {
                            level = curr->height-1;
                            SkipListElement* parent = find_left_parent(level, curr);

                            if (parent == nullptr) {
                                top_nodes[i] = curr;
                                break;
                            } else {
                                curr = parent;
                                level++;
                            }
                        } else {
                            // Some other execution claimed this ancestor
                            if (curr_update_level > level) {
                                uintE c = curr->update_level_xor;
                                while(c > level && !gbbs::atomic_compare_and_swap(&curr->update_level_xor, (uintE)c,
                                            (uintE)level))
                                    c = curr->update_level_xor;
                            }
                            top_nodes[i] = nullptr;
                            break;
                        }
                    }
            });

            parallel_for(0, new_values->size(), [&](size_t i){
                    if (top_nodes[i] != nullptr) {
                        update_top_down_xor(top_nodes[i]->height-1, top_nodes[i]);
                    }
            });
        }
    }

    void batch_update_sum(sequence<std::pair<SkipListElement*, size_t>>* new_values) {
        auto top_nodes = sequence<SkipListElement*>(new_values->size(), nullptr);
        sequence<std::pair<SkipListElement*, size_t>>&
            new_values_ref = *new_values;
        if (new_values != nullptr) {
            parallel_for(0, new_values->size(), [&](size_t i) {
                    SkipListElement* this_element = new_values_ref[i].first;
                    size_t new_size = new_values_ref[i].second;

                    this_element->size[0] = new_size;

                    size_t level = 0;
                    SkipListElement* curr = this_element;
                    while(true) {
                        uintE curr_update_level = curr->update_level_sum;
                        if (curr_update_level == UINT_E_MAX && gbbs::atomic_compare_and_swap(&curr->update_level_sum,
                                UINT_E_MAX, (uintE) level)) {
                            level = curr->height-1;
                            SkipListElement* parent = find_left_parent(level, curr);

                            if (parent == nullptr) {
                                top_nodes[i] = curr;
                                break;
                            } else {
                                curr = parent;
                                level++;
                            }
                        } else {
                            // Some other execution claimed this ancestor
                            if (curr_update_level > level) {
                                uintE c = curr->update_level_sum;
                                while(c > level && !gbbs::atomic_compare_and_swap(&curr->update_level_sum, (uintE)c,
                                            (uintE)level))
                                    c = curr->update_level_sum;
                            }
                            break;
                        }
                    }
            });

            parallel_for(0, new_values->size(), [&](size_t i){
                    if (top_nodes[i] != nullptr) {
                        update_top_down_sum(top_nodes[i]->height-1, top_nodes[i]);
                    }
            });
        }
    }

    void batch_join(sequence<std::pair<SkipListElement*, SkipListElement*>>* joins) {
            sequence<std::pair<SkipListElement*, SkipListElement*>>& joins_ref = *joins;
            auto join_lefts = sequence<std::pair<SkipListElement*,
                 sequence<sequence<std::pair<uintE, uintE>>>>>(joins->size());
            auto join_left_sizes = sequence<std::pair<SkipListElement*, size_t>>(joins->size());
            parallel_for(0, joins->size(), [&] (size_t i) {
                join(joins_ref[i].first, joins_ref[i].second);
                join_lefts[i] = std::make_pair(joins_ref[i].first, joins_ref[i].first->values[0]);
                join_left_sizes[i] = std::make_pair(joins_ref[i].first, joins_ref[i].first->size[0]);
            });

            batch_update_xor(&join_lefts);
            batch_update_sum(&join_left_sizes);
    }

    sequence<SkipListElement*> batch_split(sequence<SkipListElement*>* splits) {
            sequence<SkipListElement*>& splits_ref = *splits;
            sequence<SkipListElement*> results = sequence<SkipListElement*>(splits->size());
            parallel_for(0, splits->size(), [&](size_t i){
                results[i] = split(splits_ref[i]);
            });

            // Perform updates but only if some other thread hasn't already performed the update
            parallel_for(0, splits->size(), [&](size_t i){
                SkipListElement* curr = splits_ref[i];
                bool can_proceed = (curr->update_level_xor == UINT_E_MAX) && (curr->update_level_sum == UINT_E_MAX)
                    && gbbs::atomic_compare_and_swap(&curr->update_level_xor, UINT_E_MAX, (uintE)0)
                    && gbbs::atomic_compare_and_swap(&curr->update_level_sum, UINT_E_MAX, (uintE)0);

                if (can_proceed) {
                    sequence<sequence<std::pair<uintE, uintE>>> xor_sums = curr->values[0];
                    size_t old_size = curr->size[0];
                    size_t level = 0;
                    while(true) {
                        if(level < curr->height - 1) {
                            level++;
                            curr->values[level] = xor_sums;
                            curr->size[level] = old_size;
                        } else {
                            curr = curr->neighbors[level].first;
                            if (curr == nullptr) {
                                break;
                            } else {
                                parallel_for(0, xor_sums.size(), [&](size_t ii) {
                                        parallel_for(0, xor_sums[ii].size(), [&](size_t ij) {
                                            xor_sums[ii][ij].first ^= curr->values[level][ii][ij].first;
                                            xor_sums[ii][ij].second ^= curr->values[level][ii][ij].second;
                                        });
                                });

                                old_size -= curr->size[level];
                            }
                        }
                    }
                }
            });

            parallel_for(0, splits->size(), [&](size_t i) {
                splits_ref[i]->update_level_xor = UINT_E_MAX;
                splits_ref[i]->update_level_sum = UINT_E_MAX;
            });

            return results;
    }

    // Get the XOR of the entire sequence
    sequence<sequence<std::pair<uintE, uintE>>> get_xor(SkipListElement* this_element) {
            SkipListElement* root = find_representative(this_element);
            size_t level = root->height-1;
            sequence<sequence<std::pair<uintE, uintE>>> xor_sums = root->values[level];

            SkipListElement* curr = root->neighbors[level].second;
            while (curr != nullptr && curr != root) {
                parallel_for(0, xor_sums.size(), [&](size_t ii) {
                        parallel_for(0, xor_sums[ii].size(), [&](size_t ij) {
                            xor_sums[ii][ij].first ^= curr->values[level][ii][ij].first;
                            xor_sums[ii][ij].second ^= curr->values[level][ii][ij].second;
                        });
                });
                curr = curr->neighbors[level].second;
            }


            if (curr == nullptr) { // the list is not circular
                    curr = root;
                    while(true) {
                            while(level > 0 && curr->neighbors[level].first == nullptr) {
                                    level--;
                            }

                            if(level == 0 && curr->neighbors[level].first == nullptr)
                                break;

                            while(curr->neighbors[level].first != nullptr) {
                                curr = curr->neighbors[level].first;
                                parallel_for(0, xor_sums.size(), [&](size_t ii) {
                                        parallel_for(0, xor_sums[ii].size(), [&](size_t ij) {
                                            xor_sums[ii][ij].first ^= curr->values[level][ii][ij].first;
                                            xor_sums[ii][ij].second ^= curr->values[level][ii][ij].second;
                                        });
                                });
                            }
                    }
            }
            return xor_sums;
    }

    // Get the SUM of the entire sequence
    size_t get_sum(SkipListElement* this_element) {
            SkipListElement* root = find_representative(this_element);
            size_t level = root->height-1;
            size_t sums = root->size[level];

            SkipListElement* curr = root->neighbors[level].second;
            while (curr != nullptr && curr != root) {
                sums += curr->size[level];
                curr = curr->neighbors[level].second;
            }

            if (curr == nullptr) { // the list is not circular
                    curr = root;
                    while(true) {
                            while(level > 0 && curr->neighbors[level].first == nullptr) {
                                    level--;
                            }

                            if(level == 0 && curr->neighbors[level].first == nullptr)
                                break;

                            while(curr->neighbors[level].first != nullptr) {
                                curr = curr->neighbors[level].first;
                                sums += curr->size[level];
                            }
                    }
            }

            return sums;
    }
};

sequence<sequence<std::pair<uintE, uintE>>> default_values(uintE a, uintE b) {
        auto values_seq = sequence<sequence<std::pair<uintE, uintE>>>(2, sequence<std::pair<uintE, uintE>>(
                    ceil(log(10)/log(2)),
                    std::make_pair(a, b)));
        return values_seq;
}

inline void RunSkipList(uintE n) {
    std::cout << "Creating skip list" << std::endl;
    auto skip_list = SkipList(6);
    sequence<SkipList::SkipListElement> skip_list_neighbors = sequence<SkipList::SkipListElement>(10);

    std::cout << "creating nodes" << std::endl;
    auto curr_node = skip_list.create_node(2, nullptr, nullptr, default_values(2, 22));
    std::cout << "created first node" << std::endl;

    skip_list_neighbors[1] = curr_node;
    auto curr_node2 = skip_list.create_node(3, nullptr, nullptr, default_values(3, 33));
    skip_list_neighbors[2] = curr_node2;
    skip_list_neighbors[0] = skip_list.create_node(1, nullptr, nullptr, default_values(1, 11));
    skip_list_neighbors[3] = skip_list.create_node(4, nullptr, nullptr, default_values(4, 44));
    skip_list_neighbors[4] = skip_list.create_node(5, nullptr, nullptr, default_values(5, 55));
    skip_list_neighbors[5] = skip_list.create_node(6, nullptr, nullptr, default_values(6, 66));

    std::cout << "created nodes" << std::endl;

    std::cout << "joining nodes" << std::endl;
    sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>> join_updates
        = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(6);
    join_updates[0] = std::make_pair(&skip_list_neighbors[1], &skip_list_neighbors[2]);
    join_updates[1] = std::make_pair(&skip_list_neighbors[0], &skip_list_neighbors[1]);
    join_updates[2] = std::make_pair(&skip_list_neighbors[2], &skip_list_neighbors[0]);
    join_updates[3] = std::make_pair(&skip_list_neighbors[3], &skip_list_neighbors[4]);
    join_updates[4] = std::make_pair(&skip_list_neighbors[4], &skip_list_neighbors[3]);
    join_updates[5] = std::make_pair(&skip_list_neighbors[5], &skip_list_neighbors[5]);
    skip_list.batch_join(&join_updates);

    std::cout << "printing answers" << std::endl;

    std::cout << "node 1 height: " << skip_list_neighbors[0].height << std::endl;
    std::cout << "node 1 value: " << skip_list_neighbors[0].values[0][0][0].first << ", "
        << skip_list_neighbors[0].values[0][0][0].second << std::endl;

    std::cout << "node 2 height: " << skip_list_neighbors[1].height << std::endl;
    std::cout << "node 2 value: " << skip_list_neighbors[1].values[0][0][0].first << ", " <<
       skip_list_neighbors[1].values[0][0][0].second << std::endl;

    std::cout << "node 3 height: " << skip_list_neighbors[2].height << std::endl;
    std::cout << "node 3 value: " << skip_list_neighbors[2].values[0][0][0].first << ", " <<
       skip_list_neighbors[2].values[0][0][0].second << std::endl;

    if (skip_list_neighbors[0].neighbors[0].first != nullptr)
        std::cout << "node 1 left: " <<
            skip_list_neighbors[0].neighbors[0].first -> values[0][0][0].first <<
            skip_list_neighbors[0].neighbors[0].first -> values[0][0][0].second << std::endl;
    if (skip_list_neighbors[0].neighbors[0].second != nullptr)
        std::cout << "node 1 right: " <<
            skip_list_neighbors[0].neighbors[0].second -> values[0][0][0].first <<
            skip_list_neighbors[0].neighbors[0].second -> values[0][0][0].second << std::endl;
    if (skip_list_neighbors[1].neighbors[0].first != nullptr)
        std::cout << "node 2 left: " <<
            skip_list_neighbors[1].neighbors[0].first -> values[0][0][0].first <<
            skip_list_neighbors[1].neighbors[0].first -> values[0][0][0].second << std::endl;
    if (skip_list_neighbors[1].neighbors[0].second != nullptr)
        std::cout << "node 2 right: " <<
            skip_list_neighbors[1].neighbors[0].second -> values[0][0][0].first <<
            skip_list_neighbors[1].neighbors[0].second -> values[0][0][0].second << std::endl;
    if (skip_list_neighbors[2].neighbors[0].first != nullptr)
        std::cout << "node 3 left: " <<
            skip_list_neighbors[2].neighbors[0].first -> values[0][0][0].first <<
            skip_list_neighbors[2].neighbors[0].first -> values[0][0][0].second << std::endl;
    if (skip_list_neighbors[2].neighbors[0].second != nullptr)
        std::cout << "node 3 right: " <<
            skip_list_neighbors[2].neighbors[0].second -> values[0][0][0].first <<
            skip_list_neighbors[2].neighbors[0].second -> values[0][0][0].second << std::endl;

    std::cout << "height values" << std::endl;

    auto node_1_height = skip_list_neighbors[0].height - 1;
    auto node_2_height = skip_list_neighbors[1].height - 1;
    auto node_3_height = skip_list_neighbors[2].height - 1;

    std::cout << "printing height values" << std::endl;

    if (skip_list_neighbors[0].neighbors[node_1_height].first != nullptr) {
       std::cout << "node 1 left pointer" << std::endl;
       auto left_neighbor = skip_list_neighbors[0].neighbors[node_1_height].first;
       std::cout << "node 1 values left: " <<
           skip_list_neighbors[0].neighbors[node_1_height].first -> values[left_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[0].neighbors[node_1_height].first -> values[left_neighbor->height - 1][0][0].second
           << std::endl;
    }

    if (skip_list_neighbors[0].neighbors[node_1_height].second != nullptr) {
       auto right_neighbor = skip_list_neighbors[0].neighbors[node_1_height].second;
       std::cout << "node 1 values right: " <<
           skip_list_neighbors[0].neighbors[node_1_height].second -> values[right_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[0].neighbors[node_1_height].second -> values[right_neighbor->height - 1][0][0].second
           << std::endl;
    }

    std::cout << "node 1 done" << std::endl;

    if (skip_list_neighbors[1].neighbors[node_2_height].first != nullptr) {
       auto left_neighbor = skip_list_neighbors[1].neighbors[node_2_height].first;
       std::cout << "node 2 values left: " <<
           skip_list_neighbors[1].neighbors[node_2_height].first -> values[left_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[1].neighbors[node_2_height].first -> values[left_neighbor->height - 1][0][0].second
           << std::endl;
    }

    if (skip_list_neighbors[1].neighbors[node_2_height].second != nullptr) {
       auto right_neighbor = skip_list_neighbors[1].neighbors[node_2_height].second;
       std::cout << "node 2 height right: " <<
           skip_list_neighbors[1].neighbors[node_2_height].second -> values[right_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[1].neighbors[node_2_height].second -> values[right_neighbor->height - 1][0][0].second
           << std::endl;
    }

    if (skip_list_neighbors[2].neighbors[node_3_height].first != nullptr) {
       auto left_neighbor = skip_list_neighbors[2].neighbors[node_3_height].first;
       std::cout << "node 3 height left: " <<
           skip_list_neighbors[2].neighbors[node_3_height].first -> values[left_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[2].neighbors[node_3_height].first -> values[left_neighbor->height - 1][0][0].second
           << std::endl;
    }

    if (skip_list_neighbors[2].neighbors[node_3_height].second != nullptr) {
       auto right_neighbor = skip_list_neighbors[2].neighbors[node_3_height].second;
       std::cout << "node 3 height right: " <<
           skip_list_neighbors[2].neighbors[node_3_height].second -> values[right_neighbor->height - 1][0][0].first <<
           skip_list_neighbors[2].neighbors[node_3_height].second -> values[right_neighbor->height - 1][0][0].second
           << std::endl;
    }

    auto one_left_parent = skip_list.find_left_parent(0, &skip_list_neighbors[0]);
    if (one_left_parent != nullptr)
        std::cout << "node 1 left parent: " <<
            one_left_parent->values[0][0][0].first <<
            one_left_parent->values[0][0][0].second << std::endl;

    auto one_right_parent = skip_list.find_right_parent(0, &skip_list_neighbors[0]);
    if (one_right_parent != nullptr)
        std::cout << "node 1 right parent: " <<
            one_right_parent->values[0][0][0].first <<
            one_right_parent->values[0][0][0].second << std::endl;

    auto two_left_parent = skip_list.find_left_parent(0, &skip_list_neighbors[1]);
    if (two_left_parent != nullptr)
        std::cout << "node 2 left parent: " <<
            two_left_parent->values[0][0][0].first <<
            two_left_parent->values[0][0][0].second << std::endl;

    auto two_right_parent = skip_list.find_right_parent(0, &skip_list_neighbors[1]);
    if (two_right_parent != nullptr)
        std::cout << "node 2 right parent: " <<
            two_right_parent->values[0][0][0].first <<
            two_right_parent->values[0][0][0].second << std::endl;

    auto three_right_parent = skip_list.find_right_parent(0, &skip_list_neighbors[2]);
    if (three_right_parent != nullptr)
        std::cout << "node 3 right parent: " <<
            three_right_parent->values[0][0][0].first <<
            three_right_parent->values[0][0][0].second << std::endl;

    auto three_left_parent = skip_list.find_left_parent(0, &skip_list_neighbors[2]);
    if (three_left_parent != nullptr)
        std::cout << "node 3 left parent: " <<
            three_left_parent->values[0][0][0].first <<
            three_left_parent->values[0][0][0].second << std::endl;

    std::cout << "representative node 1: " << skip_list.find_representative(&skip_list_neighbors[0])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 2: " << skip_list.find_representative(&skip_list_neighbors[1])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 3: " << skip_list.find_representative(&skip_list_neighbors[2])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 4: " << skip_list.find_representative(&skip_list_neighbors[3])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 5: " << skip_list.find_representative(&skip_list_neighbors[4])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 6: " << skip_list.find_representative(&skip_list_neighbors[5])->values[0][0][0].first
        << std::endl;

    std::cout << "total xor subtree 1: "
        << skip_list.get_xor(&skip_list_neighbors[0])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[0])[0][0].second
        << "; "
        << skip_list.get_xor(&skip_list_neighbors[1])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[1])[0][0].second
        << "; "
        << skip_list.get_xor(&skip_list_neighbors[2])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[2])[0][0].second
        << std::endl;

    std::cout << "total size subtree 1: "
        << skip_list.get_sum(&skip_list_neighbors[0])
        << skip_list.get_sum(&skip_list_neighbors[0])
        << "; "
        << skip_list.get_sum(&skip_list_neighbors[1]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[1])
        << "; "
        << skip_list.get_sum(&skip_list_neighbors[2]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[2])
        << std::endl;

    std::cout << "total sum subtree 2: "
        << skip_list.get_xor(&skip_list_neighbors[3])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[3])[0][0].second << "; "
        << skip_list.get_xor(&skip_list_neighbors[4])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[4])[0][0].second
        << std::endl;

    std::cout << "total size subtree 2: "
        << skip_list.get_sum(&skip_list_neighbors[3]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[3]) << "; "
        << skip_list.get_sum(&skip_list_neighbors[4]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[4])
        << std::endl;

    std::cout << "total sum subtree 3: "
        << skip_list.get_xor(&skip_list_neighbors[5])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[5])[0][0].second
        << std::endl;

    std::cout << "total size subtree 3: "
        << skip_list.get_sum(&skip_list_neighbors[5]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[5])
        << std::endl;

    std::cout << "splitting some nodes" << std::endl;
    sequence<SkipList::SkipListElement*> splits
        = sequence<SkipList::SkipListElement*>(4);
    splits[0] = &skip_list_neighbors[2];
    splits[1] = &skip_list_neighbors[4];
    splits[2] = &skip_list_neighbors[5];
    splits[3] = &skip_list_neighbors[1];
    skip_list.batch_split(&splits);

    std::cout << "representative node 1: " << skip_list.find_representative(&skip_list_neighbors[0])->values[0][0][0].second
        << std::endl;
    std::cout << "representative node 2: " << skip_list.find_representative(&skip_list_neighbors[1])->values[0][0][0].second
        << std::endl;
    std::cout << "representative node 3: " << skip_list.find_representative(&skip_list_neighbors[2])->values[0][0][0].second
        << std::endl;
    std::cout << "representative node 4: " << skip_list.find_representative(&skip_list_neighbors[3])->values[0][0][0].second
        << std::endl;
    std::cout << "representative node 5: " << skip_list.find_representative(&skip_list_neighbors[4])->values[0][0][0].second
        << std::endl;
    std::cout << "representative node 6: " << skip_list.find_representative(&skip_list_neighbors[5])->values[0][0][0].second
        << std::endl;

    std::cout << "total sum subtree 1: " << skip_list.get_xor(&skip_list_neighbors[0])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[1])[0][0].first << std::endl;

    std::cout << "total size subtree 1: " << skip_list.get_sum(&skip_list_neighbors[0]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[1]) << std::endl;

    std::cout << "total sum subtree 2: "  << skip_list.get_xor(&skip_list_neighbors[2])[0][0].first
        << std::endl;

    std::cout << "total size subtree 2: "  << skip_list.get_sum(&skip_list_neighbors[2])
        << std::endl;

    std::cout << "total sum subtree 3: " << skip_list.get_xor(&skip_list_neighbors[3])[0][0].second << ", "
        << skip_list.get_xor(&skip_list_neighbors[4])[0][0].second
        << std::endl;

    std::cout << "total size subtree 3: " << skip_list.get_sum(&skip_list_neighbors[3]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[4])
        << std::endl;

    std::cout << "total sum subtree 6: " << skip_list.get_xor(&skip_list_neighbors[5])[0][0].first << std::endl;
    std::cout << "total size subtree 6: " << skip_list.get_sum(&skip_list_neighbors[5]) << std::endl;

    std::cout << "joining more nodes" << std::endl;
    sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>> join_updates_1
        = sequence<std::pair<SkipList::SkipListElement*, SkipList::SkipListElement*>>(3);
    join_updates_1[0] = std::make_pair(&skip_list_neighbors[1], &skip_list_neighbors[2]);
    join_updates_1[1] = std::make_pair(&skip_list_neighbors[5], &skip_list_neighbors[0]);
    join_updates_1[2] = std::make_pair(&skip_list_neighbors[2], &skip_list_neighbors[5]);
    skip_list.batch_join(&join_updates_1);

    std::cout << "representative node 1: " << skip_list.find_representative(&skip_list_neighbors[0])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 2: " << skip_list.find_representative(&skip_list_neighbors[1])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 3: " << skip_list.find_representative(&skip_list_neighbors[2])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 4: " << skip_list.find_representative(&skip_list_neighbors[3])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 5: " << skip_list.find_representative(&skip_list_neighbors[4])->values[0][0][0].first
        << std::endl;
    std::cout << "representative node 6: " << skip_list.find_representative(&skip_list_neighbors[5])->values[0][0][0].first
        << std::endl;

    std::cout << "total sum subtree 1: " << skip_list.get_xor(&skip_list_neighbors[0])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[1])[0][0].first
        << ", " << skip_list.get_xor(&skip_list_neighbors[2])[0][0].first
        << ", " << skip_list.get_xor(&skip_list_neighbors[5])[0][0].first << std::endl;

    std::cout << "total XOR subtree 2: " << skip_list.get_xor(&skip_list_neighbors[0])[0][0].second << ", "
        << skip_list.get_xor(&skip_list_neighbors[1])[0][0].second
        << ", " << skip_list.get_xor(&skip_list_neighbors[2])[0][0].second
        << ", " << skip_list.get_xor(&skip_list_neighbors[5])[0][0].second << std::endl;

    std::cout << "total size subtree 1: " << skip_list.get_sum(&skip_list_neighbors[0]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[1])
        << ", " << skip_list.get_sum(&skip_list_neighbors[2])
        << ", " << skip_list.get_sum(&skip_list_neighbors[5]) << std::endl;

    std::cout << "total sum subtree 2: " << skip_list.get_xor(&skip_list_neighbors[3])[0][0].first << ", "
        << skip_list.get_xor(&skip_list_neighbors[4])[0][0].first
        << std::endl;

    std::cout << "total XOR subtree 2: " << skip_list.get_xor(&skip_list_neighbors[3])[0][0].second << ", "
        << skip_list.get_xor(&skip_list_neighbors[4])[0][0].second
        << std::endl;

    std::cout << "total size subtree 2: " << skip_list.get_sum(&skip_list_neighbors[3]) << ", "
        << skip_list.get_sum(&skip_list_neighbors[4])
        << std::endl;


}

}  // namespace gbbs
