"""
Evaluation functions for explanations generated through methods that PyG provides.
Evaluates fidelity either on a whole model or on top-k percentages of the most impactful edges.
Fidelity+: How much the model prediction changes if the most important features are removed (i.e. higher is better)
Fidelity-: How much the model prediction changes when only the most important features are kept (i.e. lower is better)
"""

import json
from pathlib import Path
from statistics import mean

import torch
from torch_geometric.explain.metric import fidelity


def evaluate_fidelity(results, only_correct=True):
    """Evaluate generated explanations with PyG's fidelity metric."""
    if only_correct:
        results = [
            result
            for result in results
            if result["predicted_class"] == int(result["data"].y.item())
        ]

    if not results:
        raise ValueError("No explanations are available for fidelity evaluation.")

    scores = []
    for result in results:
        # fidelity applies the mask to the input graph and computes a score based on the masked model's predictions
        # note: fidelity is explained in the documentation as using a subgraph, but really it uses continuous explanation values
        # which creates a subgraph implicitly because only important values will have influence 
        # actual bounded subgraphs (e.g. only top-k features) need preprocessing 
        fidelity_plus, fidelity_minus = fidelity(
            result["explainer"],
            result["explanation"],
        )
        scores.append({
            "index": int(result["index"]),
            "fidelity_plus": fidelity_plus,  # plus: how much the prediction changes when important features are removed (realized by inverting the explanation mask)
            "fidelity_minus": fidelity_minus, # minus: how much the prediction changes when only keeping important features (applying the explanation mask)
        })

    return {
        "scores": scores,
        "graph_count": len(scores),
        "mean_fidelity_plus": mean(score["fidelity_plus"] for score in scores),
        "mean_fidelity_minus": mean(score["fidelity_minus"] for score in scores),
    }


def create_topk_mask(result, fraction, hard=False):
    """
    Return a copy of an explanation containing only its top-k edge values.
    If hard is set to True, all kept values will be set to 1, otherwise the given explanation values are kept.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Top-k fraction must be greater than 0 and at most 1.")

    explanation = result["explanation"].clone()
    edge_mask = explanation.edge_mask
    k = max(1, int(fraction * edge_mask.numel()))
    top_k = torch.topk(edge_mask.abs(), k).indices

    new_mask = torch.zeros_like(edge_mask)
    new_mask[top_k] = 1.0 if hard else edge_mask[top_k]
    explanation.edge_mask = new_mask
    return explanation


def evaluate_fidelity_topk(results, step, only_correct=True, hard=False):
    """Evaluate fidelity after retaining top edges at percentage steps."""
    if not 1 <= step <= 100:
        raise ValueError("The top-k step must be an integer between 1 and 100.")

    evaluations = []
    for percentage in range(step, 101, step):
        fraction = percentage / 100
        topk_results = [
            {**result, "explanation": create_topk_mask(result, fraction, hard=hard)}
            for result in results
        ]
        evaluation = evaluate_fidelity(topk_results, only_correct=only_correct)
        evaluations.append({
            "percentage": percentage,
            "fraction": fraction,
            **evaluation,
        })

    return {"evaluations": evaluations}


def save_fidelity_data(
    evaluation,
    output_root="output/explanations",
    filename="fidelity.json",
):
    """Save fidelity evaluation results as JSON."""
    output_folder = Path(output_root)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / filename
    output_file.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return output_file

