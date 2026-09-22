from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm import DummyExplainer, PGExplainer

from train_demo_gnn import GIN

ATOM_NAMES = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br",
}

CLASS_MEANINGS = {
    0: "non-mutagenic",
    1: "mutagenic",
}


def atom_symbol_from_index(index):
    return ATOM_NAMES.get(int(index), f"X{int(index)}")


def load_model(checkpoint_path="gin_mutag.pt"):
    """load a pretrained GIN model from a checkpoint and return it and the device it's on"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["model_config"]

    model = GIN(
        num_features=config["num_features"],
        hidden_channels=config["hidden_channels"],
        num_classes=config["num_classes"],
        num_layers=config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def build_GNNexplainer(model, explanation_type):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type=explanation_type,
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )

def build_DummyExplainer(model, explanation_type):
    return Explainer(
        model=model,
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )


def build_PGExplainer(model, explanation_type):
    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type=explanation_type,
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )


def _train_pgexplainer(explainer, model, dataset, indices, device, explanation_type):
    for epoch in range(explainer.algorithm.epochs):
        for index in indices:
            data = dataset[index].to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            if explanation_type == "phenomenon":
                target = data.y
            else:
                with torch.no_grad():
                    target = model(data.x, data.edge_index, batch).argmax(dim=-1)

            explainer.algorithm.train(
                epoch,
                model,
                data.x,
                data.edge_index,
                target=target,
                batch=batch,
            )


def run_explanations(indices=None, algorithm="GNNExplainer", explanation_type="model", dataset_path="data/MUTAG"):
    if explanation_type not in {"model", "phenomenon"}:
        raise ValueError("explanation_type must be 'model' or 'phenomenon'")
    if algorithm == "PGExplainer" and explanation_type != "phenomenon":
        raise ValueError("PGExplainer only supports phenomenon explanations in this PyG version.")

    dataset = TUDataset(root=dataset_path, name="MUTAG")
    model, device = load_model()
    if algorithm == "GNNExplainer":
        explainer = build_GNNexplainer(model, explanation_type)
    elif algorithm == "Dummy":
        explainer = build_DummyExplainer(model, explanation_type)
    elif algorithm == "PGExplainer":
        explainer = build_PGExplainer(model, explanation_type)
    else:
        raise ValueError("Unknown explainer algorithm defined.")
    selected_indices = list(range(len(dataset)) if indices is None else indices)
    if algorithm == "PGExplainer":
        _train_pgexplainer(
            explainer,
            model,
            dataset,
            selected_indices,
            device,
            explanation_type,
        )
    results = []

    for index in selected_indices:
        if index % 10 == 0:
            print(f"Running explanation for graph index {index}...")
        data = dataset[index].to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        with torch.no_grad():
            prediction = model(data.x, data.edge_index, batch)
        predicted_class = prediction.argmax(dim=-1).item()

        explanation_kwargs = {
            "x": data.x,
            "edge_index": data.edge_index,
            "batch": batch,
        }
        if explanation_type == "phenomenon":
            explanation_kwargs["target"] = data.y
        elif algorithm == "PGExplainer":
            explanation_kwargs["target"] = prediction.argmax(dim=-1)

        explanation = explainer(**explanation_kwargs)
        results.append({
            "data": data,
            "device": device,
            "index": index,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "explanation": explanation,
            "explainer": explainer,
            "explanation_type": explanation_type,
        })

    return results


def run_explanation(
    index,
    algorithm="GNNExplainer",
    explanation_type="model",
    dataset_path="data/MUTAG",
):
    return run_explanations(
        indices=[index],
        algorithm=algorithm,
        explanation_type=explanation_type,
        dataset_path=dataset_path,
    )[0]


def visualization_path(index, true_class, explanation_type, output_root="output"):
    output_folder = CLASS_MEANINGS.get(int(true_class), "unknown")
    output_path = Path(output_root) / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"mutag_explanation_{index}_{output_folder}_{explanation_type}.png"
