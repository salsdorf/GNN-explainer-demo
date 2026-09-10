from pathlib import Path

import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.explain import Explainer, GNNExplainer

from train_demo_gnn import GIN

# atom types from MUTAG readme
ATOM_NAMES = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br",
}

# target: standard 0-1 classification for mutagenicism
CLASS_MEANINGS = {
    0: "non-mutagenic",
    1: "mutagenic",
}


def atom_symbol_from_index(index):
    return ATOM_NAMES.get(int(index), f"X{int(index)}")


# use gpu if available
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = TUDataset(
    root="data/MUTAG",
    name="MUTAG",
)

# load checkpoint trained by train_demo_gnn
checkpoint = torch.load(
    "gin_mutag.pt",
    map_location=DEVICE,
)

config = checkpoint["model_config"]

#reinstantiate model 
model = GIN(
    num_features=config["num_features"],
    hidden_channels=config["hidden_channels"],
    num_classes=config["num_classes"],
    num_layers=config["num_layers"],
).to(DEVICE)

# set weights from checkpoint
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# set to evaluation mode
model.eval()


# Select one graph to test on (the first one)
# todo: select others?

index = 0
data = dataset[index].to(DEVICE)

#batch: tell the model which nodes belong to which graph, here only one graph so all zeros
batch = torch.zeros(
    data.num_nodes,
    dtype=torch.long,
    device=DEVICE,
)

#predict on selected graph 

# no grad because irrelevant outside of training and faster
with torch.no_grad():

    prediction = model(
        data.x,
        data.edge_index,
        batch,
    )

# todo: what are logits
print("Prediction logits:", prediction)
print("True label:", data.y.item(), "->", CLASS_MEANINGS.get(int(data.y.item()), "unknown"))

predicted_class = prediction.argmax(dim=-1).item()
print(
    "Predicted class:",
    predicted_class,
    "->",
    CLASS_MEANINGS.get(predicted_class, "unknown"),
)

print()
print("Selected graph atom types:")
node_atom_indices = data.x.argmax(dim=-1).tolist()
for node_index, atom_index in enumerate(node_atom_indices):
    print(f"  Node {node_index}: {atom_symbol_from_index(atom_index)}")


# Build explainer

explainer = Explainer(
    #todo: try phenomenon too

    model=model,

    # metadata for learning masks
    algorithm=GNNExplainer(
        epochs=200,
        lr=0.01,
    ),

    explanation_type="model",
    node_mask_type="attributes",    # learn mask based on node attributes i.e. features
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="graph",     # prediction level
        return_type="raw",      # return logits raw, todo try different options
    ),
)

# run explainer on selected graph
explanation = explainer(
    data.x,
    data.edge_index,
    batch=batch,
)

# todo save this somehow
# print full explanation object
print()
print("Explanation:")
print(explanation)

print()
print("Node mask shape:")
print(explanation.node_mask.shape)

print()
print("Edge mask shape:")
print(explanation.edge_mask.shape)

# shows edge importance as table
print()
print("Edge mask:")
print(explanation.edge_mask)

print()
print("Top important atoms for this prediction:")
node_importance = explanation.node_mask.abs().mean(dim=-1)
for rank, idx in enumerate(torch.argsort(node_importance, descending=True)[: min(5, node_importance.numel())].tolist(), start=1):
    atom_index = int(data.x[idx].argmax(dim=-1).item())
    atom_name = atom_symbol_from_index(atom_index)
    score = node_importance[idx].item()
    print(f"  #{rank}: Node {idx} ({atom_name}) -> importance {score:.4f}")

print()
print("Top important bonds for this prediction:")
edge_importance = explanation.edge_mask
for rank, idx in enumerate(torch.argsort(edge_importance, descending=True)[: min(5, edge_importance.numel())].tolist(), start=1):
    src, dst = data.edge_index[:, idx].tolist()
    src_atom = atom_symbol_from_index(data.x[src].argmax(dim=-1).item())
    dst_atom = atom_symbol_from_index(data.x[dst].argmax(dim=-1).item())
    score = edge_importance[idx].item()
    print(f"  #{rank}: {src_atom}-{dst_atom} (edge {idx}) -> importance {score:.4f}")

print()
print("Interpretation of classes:")
for cls_id, meaning in CLASS_MEANINGS.items():
    print(f"  Class {cls_id}: {meaning}")

print()
print("Meaning of this graph:")
print(
    f"  The selected molecule is classified as {CLASS_MEANINGS.get(predicted_class, 'unknown')} "
    f"because the model assigned the highest score to class {predicted_class}."
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    node_importance = explanation.node_mask.abs().mean(dim=-1)
    atom_indices = data.x.argmax(dim=-1).tolist()

    for node_idx, atom_idx in enumerate(atom_indices):
        atom_name = atom_symbol_from_index(atom_idx)
        importance = float(node_importance[node_idx].item())
        g.add_node(node_idx, atom=atom_name, importance=importance)

    for edge_idx in range(data.edge_index.size(1)):
        src, dst = data.edge_index[:, edge_idx].tolist()
        importance = float(explanation.edge_mask[edge_idx].item())
        g.add_edge(int(src), int(dst), weight=importance)

    pos = nx.spring_layout(g, seed=42)
    node_colors = [float(g.nodes[n]["importance"]) for n in g.nodes()]
    edge_widths = [max(1.0, float(g.edges[e]["weight"]) * 6.0) for e in g.edges()]
    edge_colors = [float(g.edges[e]["weight"]) for e in g.edges()]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(
        g,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.viridis,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=[500 + 300 * max(0.0, v) for v in node_colors],
        cmap=plt.cm.viridis,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels={n: g.nodes[n]["atom"] for n in g.nodes()},
        font_size=12,
        ax=ax,
    )

    importance_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm_importance = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=importance_norm)
    sm_importance.set_array([])

    cbar = fig.colorbar(sm_importance, ax=ax, fraction=0.045, pad=0.1)
    #cbar.set_label('Importance', rotation=270, labelpad=18)
    cbar.ax.set_title('Importance', pad=8)

    ax.set_title(
        f"MUTAG graph explanation\n"
        f"Predicted class: {predicted_class} ({CLASS_MEANINGS.get(predicted_class, 'unknown')})"
    )
    output_folder = {
        0: "nonmutagenic",
        1: "mutagenic",
    }.get(int(data.y.item()), "unknown")
    output_path = Path("output") / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    visualization_path = output_path / f"mutag_explanation_graph_{index}.png"

    plt.tight_layout()
    plt.savefig(visualization_path, bbox_inches='tight', dpi=200)
    print()
    print(f"Saved visualization to: {visualization_path}")
except Exception as exc:
    print()
    print("Visualization skipped because plotting dependencies are unavailable or failed:", exc)
