from typing import Dict, Optional, Set, List, Tuple, Union
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from pygraphviz import AGraph

from step_py.dyndim import DynDim
from step_py.ops import StepOps


def shape_to_str(
    shape: Tuple[Union[int, DynDim], ...], dyndim_max_length: int = 10
) -> str:
    str_shape = []
    for dim in shape:
        if isinstance(dim, DynDim):
            """Truncate string to max_length characters, adding '...' if truncated."""
            dim_str = str(dim.expr)
            if len(dim_str) <= dyndim_max_length:
                str_shape.append(dim_str)
            else:
                str_shape.append(dim_str[: dyndim_max_length - 3] + "...")
        else:
            str_shape.append(dim)

    return str(str_shape)


def save_graph_format(
    digraph: nx.DiGraph,
    output_filename: str,
    format: List[str],
    subgraph_nodes: Optional[List] = None,  # nodes to highlight as a subgraph
):
    print(f"Visualizing the STEP GRAPH...")

    agraph: AGraph = to_agraph(digraph)

    class_color_map = {
        "OffChipLoad": "gray",
        "OffChipStore": "darkgray",
        "BinaryMap": "darkcyan",
        "BinaryMapAccum": "darkcyan",
        # "ScanInclusive": "lightseagreen",
        # "ScanExclusive": "lightseagreen",
        "Accum": "turquoise",
        # "ExpandMap": "cyan",
        # "Zip": "lemonchiffon",
        "Broadcast": "lemonchiffon",
        "Bufferize": "orange",
        # "Parallelize": "goldenrod",
        "Promote": "gold",
        # "Enumerate": "gold",
        "Flatten": "gold",
        # "Reshape": "gold",
        "RepeatStatic": "yellow",
        # "RepeatRef": "yellow",
        # "RepeatRefRank": "yellow",
        "FlatPartition": "lightcoral",
        "FlatReassemble": "indianred",
    }

    for node in digraph.nodes(data=True):
        node_id: StepOps = node[0]
        class_name = node_id.__class__.__name__

        n = agraph.get_node(node_id)
        n.attr["shape"] = "box"  # Set the node shape to a rectangle
        n.attr["style"] = "rounded,filled"  # Add rounded corners and fill color
        n.attr["fillcolor"] = (
            class_color_map[class_name] if class_name in class_color_map else "white"
        )  # Set background color
        if class_name in ["OffChipStore", "PrinterContext", "ConsumerContext"]:
            n.attr["label"] = str(node_id)
        elif class_name in ["Broadcast", "FlatPartition", "Parallelize"]:
            n.attr["label"] = "\n".join(
                [
                    str(node_id),
                    "-------------",
                    "output stream shape:",
                ]
                + [shape_to_str(out_i.shape) for out_i in node_id.stream_list]
                + ["data type:", str(node_id.stream_list[0].stream_dtype)]
            )
        elif class_name in ["EagerMerge"]:
            n.attr["label"] = "\n".join(
                [
                    str(node_id),
                    "-------------",
                    "output stream shape:",
                    f"data: {shape_to_str(node_id.stream_idx(0).shape)}",
                    f"sel: {shape_to_str(node_id.stream_idx(1).shape)}",
                    "data type:",
                    str(node_id.stream_idx(0).stream_dtype),
                    "sel type:",
                    str(node_id.stream_idx(1).stream_dtype),
                ]
            )
        else:
            n.attr["label"] = "\n".join(
                [
                    str(node_id),
                    "-------------",
                    "output stream shape:",
                    shape_to_str(node_id.stream.shape),
                    "data type:",
                    str(node_id.stream.stream_dtype),
                ]
            )

    if subgraph_nodes is not None:
        subgraph = agraph.add_subgraph(
            subgraph_nodes,
            name="cluster_preserved",
            label="Preserved Nodes",
            color="gray",
            bgcolor="lightgray",
            style="filed,dashed",
        )

    if "png" in format:
        agraph.draw(f"{output_filename}.png", prog="dot", format="png")
        print(f"finished writing the STEP GRAPH to {output_filename}.png")
    if "svg" in format:
        agraph.draw(f"{output_filename}.svg", prog="dot", format="svg")
        print(f"finished writing the STEP GRAPH to {output_filename}.svg")

    print(f"finished writing the STEP GRAPH to {output_filename}")
