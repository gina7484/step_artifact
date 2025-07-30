from typing import Dict, List
import networkx as nx
from step_py.ops import *

# from step.datatype.stream import StreamBundleTP
# from step.utils.error import TypeMismatchError


def infer_broadcast(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # Collect nodes with more than 1 outgoing edge and their destination nodes
    broadcast_nodes: Dict[StepOps, List[StepOps]] = {}

    for node, neighbors in graph.adjacency():
        if len(neighbors) > 1:
            neighbor_nodes = list(neighbors.keys())
            broadcast_nodes[node] = neighbor_nodes

            multi_edge_nodes = []
            for neighbor in neighbor_nodes:
                edge_count = graph.number_of_edges(node, neighbor)
                if edge_count > 1:
                    multi_edge_nodes.extend([neighbor] * (edge_count - 1))

            broadcast_nodes[node].extend(multi_edge_nodes)

        elif len(neighbors) == 1:
            edge_count = graph.number_of_edges(node, list(neighbors.keys())[0])
            if edge_count > 1:
                # Handling multi-edge between two nodes
                broadcast_nodes[node] = list(neighbors.keys()) * edge_count

    edges_to_remove = []
    edges_to_add = []

    for node, dst_node_list in broadcast_nodes.items():

        if isinstance(node, (FlatPartition, EagerMerge, Parallelize)):
            if node.num_consumers < len(dst_node_list):
                # This means there is broadcast happening in some of the output streams
                # [TODO] Fix this for BinaryMap and DynStreamify
                regrouped_dst_node_list = [[] for _ in range(node.num_consumers)]
                for dst_node in dst_node_list:
                    input_list = dst_node.input_list
                    for input_to_dst_node in input_list:
                        if isinstance(input_to_dst_node, tuple):
                            src_of_dst_node, src_idx_of_dst_node = input_to_dst_node
                            if src_of_dst_node == node:
                                regrouped_dst_node_list[src_idx_of_dst_node].append(
                                    dst_node
                                )

                dst_node_set = set(dst_node_list)

                incoming_edge_per_dst_node = {}
                for unique_dst_node in dst_node_set:
                    incoming_edge_per_dst_node[unique_dst_node] = graph.number_of_edges(
                        node, unique_dst_node
                    )

                lowest_existing_edge_key = {}
                for key, value in incoming_edge_per_dst_node.items():
                    lowest_existing_edge_key[key] = 0

                for idx, dst_node_list_i in enumerate(regrouped_dst_node_list):

                    if len(dst_node_list_i) > 1:
                        src_node = (node, idx)  # type: ignore (Cannot infer Union[EagerMerge, FlatPartition] is a subclass of StepOps)
                        broadcast = Broadcast(
                            graph=graph,
                            input=src_node,
                            num_consumers=len(dst_node_list_i),
                        )

                        for broadcast_idx, dst_node in enumerate(dst_node_list_i):
                            if incoming_edge_per_dst_node[dst_node] == 1:
                                edges_to_remove.append((node, dst_node))
                            else:
                                assert (
                                    lowest_existing_edge_key[dst_node]
                                    < incoming_edge_per_dst_node[dst_node]
                                )
                                edges_to_remove.append(
                                    (node, dst_node, lowest_existing_edge_key[dst_node])
                                )
                                lowest_existing_edge_key[dst_node] += 1
                            dst_node.replace_input(
                                org_input=src_node, new_input=(broadcast, broadcast_idx)
                            )
                            edges_to_add.append((broadcast, dst_node))

                    else:
                        continue
            else:
                continue
        elif isinstance(node, Broadcast):
            continue
        else:
            assert isinstance(node, StepOps)
            src_node: StepOps = node

            broadcast = Broadcast(
                graph=graph, input=src_node, num_consumers=len(dst_node_list)
            )
            # print(
            #     f"Broadcasting {src_node} to {len(dst_node_list)} consumers: {dst_node_list}"
            # )

            # Update the downstream nodes
            for idx, dst_node in enumerate(dst_node_list):
                edges_to_remove.append((node, dst_node))
                dst_node.replace_input(org_input=src_node, new_input=(broadcast, idx))
                edges_to_add.append((broadcast, dst_node))

    # # Propagate the changes into the graph
    graph.remove_edges_from(edges_to_remove)  # Remove the edges
    graph.add_edges_from(edges_to_add)  # Add new edges

    return graph
