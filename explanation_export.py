import json
from pathlib import Path

from explanation_runner import atom_symbol_from_index


BOND_TYPES = {
    0: "aromatic",
    1: "single",
    2: "double",
    3: "triple",
}


def _apply_limit(items, limit):
    if limit < 0:
        raise ValueError("Limits for exported nodes and edges cannot be negative.")
    return items if limit == 0 else items[:limit]


def _top_nodes(result, limit):
    data = result["data"]
    node_mask = result["explanation"].node_mask.detach().cpu()
    node_values = node_mask.abs().mean(dim=-1)
    node_features = data.x.argmax(dim=-1).detach().cpu().tolist()

    ranked_indices = _apply_limit(
        node_values.argsort(descending=True).tolist(),
        limit,
    )
    return [
        {
            "node_index": int(node_index),
            "importance": float(node_values[node_index]),
            "atom": atom_symbol_from_index(node_features[node_index]),
        }
        for node_index in ranked_indices
    ]


def _top_edges(result, limit):
    data = result["data"]
    edge_index = data.edge_index.detach().cpu()
    edge_attr = data.edge_attr.detach().cpu()
    edge_values = result["explanation"].edge_mask.detach().cpu()
    node_features = data.x.argmax(dim=-1).detach().cpu().tolist()

    unique_edges = {}
    for edge_position in range(edge_index.size(1)):
        source = int(edge_index[0, edge_position])
        target = int(edge_index[1, edge_position])
        edge = tuple(sorted((source, target)))
        importance = float(edge_values[edge_position])

        current = unique_edges.get(edge)
        if current is None or abs(importance) > abs(current["value"]):
            bond_index = int(edge_attr[edge_position].argmax())
            unique_edges[edge] = {
                "source": edge[0],
                "target": edge[1],
                "source_atom": atom_symbol_from_index(node_features[edge[0]]),
                "target_atom": atom_symbol_from_index(node_features[edge[1]]),
                "bond_type": BOND_TYPES.get(bond_index, "unknown"),
                "value": importance,
            }

    ranked_edges = sorted(
        unique_edges.values(),
        key=lambda edge: abs(edge["value"]),
        reverse=True,
    )
    ranked_edges = _apply_limit(ranked_edges, limit)
    for edge in ranked_edges:
        edge["importance"] = abs(edge["value"])
    return ranked_edges


def _build_graph_export(result, top_nodes, top_edges):
    data = result["data"]
    return {
        "graph_index": int(result["index"]),
        "true_class": int(data.y.item()),
        "predicted_class": int(result["predicted_class"]),
        "nodes": _top_nodes(result, top_nodes),
        "edges": _top_edges(result, top_edges),
    }


def save_explanation_data(
    results,
    top_nodes=0,
    top_edges=0,
    output_root="output/explanations",
):
    explanation_type = results[0]["explanation_type"]
    output_folder = Path(output_root)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "explanations.json"

    export = {"model": [], "phenomenon": []}
    if output_file.exists():
        export.update(json.loads(output_file.read_text(encoding="utf-8")))

    export[explanation_type] = [
        _build_graph_export(result, top_nodes, top_edges)
        for result in results
    ]
    output_file.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return output_file