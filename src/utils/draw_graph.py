from typing import Dict, Optional, Set, List
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from pygraphviz import AGraph

from step_py.ops import StepOps


def save_graph_format(
    digraph: nx.DiGraph,
    output_filename: str,
    format: List[str],
    subgraph_nodes: Optional[List] = None,  # nodes to highlight as a subgraph
):
    agraph: AGraph = to_agraph(digraph)

    class_color_map = {
        "OffChipLoad": "gray",
        "OffChipStore": "darkgray",
        "BinaryMap": "darkcyan",
        # "ScanInclusive": "lightseagreen",
        # "ScanExclusive": "lightseagreen",
        # "Accum": "turquoise",
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
        if class_name in ["OffChipStore", "PrinterContext"]:
            n.attr["label"] = str(node_id)
        elif class_name in ["Broadcast", "FlatPartition", "Parallelize"]:
            n.attr["label"] = "\n".join(
                [
                    str(node_id),
                    "-------------",
                    str([out_i.shape for out_i in node_id.stream_list]),
                ]
            )
        else:
            n.attr["label"] = "\n".join(
                [str(node_id), "-------------", str(node_id.stream.shape)]
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
    if "svg" in format:
        agraph.draw(f"{output_filename}.svg", prog="dot", format="svg")

    print(f"finished writing the lowered STEP GRAPH to {output_filename}")
