"""Root file of the explainer, for calling and defining output."""

import argparse
from pathlib import Path

from explanation_runner import (
    CLASS_MEANINGS,
    run_explanations,
)
from explanation_export import save_explanation_data
from explanation_evaluation import evaluate_fidelity, evaluate_fidelity_topk, save_fidelity_data
from visualize_explanation import save_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Explain a MUTAG graph with GNNExplainer.")
    parser.add_argument(
        "--mode",
        choices=("model", "phenomenon"),
        default="model",
        help="Specifies whether to explain the model as a whole or a specific classification.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("GNNExplainer", "PGExplainer", "Dummy"),
        default="GNNExplainer",
        help="Explainer algorithm to use.",
    )
    parser.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="Dataset indices of the graph to explain.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run explanation for every graph in the provided dataset.",
    )
    parser.add_argument(
        "--m_png",
        action="store_true",
        help="Output a visualisation of the explanation of the investigated molecule graphs as a png file.",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=0,
        help="Number of most important nodes to save; 0 saves all nodes.",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=0,
        help="Number of most important edges to save; 0 saves all edges.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/explanations",
        help="Directory for exported explanation JSON files.",
    )
    parser.add_argument(
        "--save_explanation",
        action="store_true",
        help="Whether to save the generated explanation data as json. Set --top_nodes/edges to define how much data to save.",
    )
    parser.add_argument(
        "--evaluate",
        nargs="?",
        const="basic",
        choices=("basic", "topk"),
        help="Define whether and how to evaluate the explanation - Either on the full model or testing different top-k sparsity percentages.",
    )
    parser.add_argument(
        "--topk_step",
        type=int,
        default=20,
        help="Step size for evaluating different top-k-percentages of edge explanations.",
    )

    return parser.parse_args()


def print_result(result):
    data = result["data"]
    prediction = result["prediction"]
    predicted_class = result["predicted_class"]
    explanation = result["explanation"]

    print("Explanation type:", result["explanation_type"])
    print("Prediction logits:", prediction)
    print("True label:", data.y.item(), "->", CLASS_MEANINGS.get(int(data.y.item()), "unknown"))
    print("Predicted class:", predicted_class, "->", CLASS_MEANINGS.get(predicted_class, "unknown"))

    print("\nExplanation:")
    #print(explanation)
    print("\nNode mask shape:", explanation.node_mask.shape)
    print("Edge mask shape:", explanation.edge_mask.shape)
    print("Edge mask:")
    print(explanation.edge_mask)


def main():
    args = parse_args()
    if args.index is not None and args.all:
        raise ValueError("Cannot specify both --index and --all flags.")

    indices = None if args.all else (args.index or [0])
    if (not indices or len(indices) > 1) and not args.evaluate and not args.save_explanation and not args.m_png:
        print("Not enough parameters specified. Running this wouldn't do anything. Add parameters or speciufy a singular index to print result immediately.")
        print("Aborting...")
        return

    output_dir = Path(args.output_dir) / args.algorithm

    results = run_explanations(
        indices=indices,
        algorithm=args.algorithm,
        explanation_type=args.mode,
    )

    if args.evaluate:
        # ONLY_CORRECT: Whether to only investigate graphs the full model predicted correctly.
        # This can make sense for evaluating model-based explanations.
        if args.mode == "phenomenon":
            ONLY_CORRECT = False
        else:
            ONLY_CORRECT = True
        filename = "fidelity_" + args.mode
        if args.evaluate == "basic":
            evaluation = evaluate_fidelity(results, ONLY_CORRECT)
            evaluation_file = save_fidelity_data(evaluation, output_dir, filename+".json")
            print(f"Saved fidelity evaluation to: {evaluation_file}")

        elif args.evaluate == "topk":
            HARD = True # sets top-k explanation values to 1
            evaluation = evaluate_fidelity_topk(results, args.topk_step, only_correct=ONLY_CORRECT, hard=HARD)
            evaluation_file = save_fidelity_data(
                evaluation,
                output_dir,
                f"{filename}_topk_{args.topk_step}.json",
            )
            print(f"Saved top-k fidelity evaluation to: {evaluation_file}")
        else: 
            raise ValueError("Unknown --evaluate value")

    if args.save_explanation:
        output_file = save_explanation_data(
            results,
            top_nodes=args.top_nodes,
            top_edges=args.top_edges,
            output_root=output_dir,
        )
        print(f"Saved {len(results)} explanation objects to: {output_file}")

    if len(results) == 1:
        print_result(results[0])

    if args.m_png:
        for result_number, result in enumerate(results, start=1):
            try:
                output_file = save_visualization(result, output_root=output_dir)
                print(f"[{result_number}/{len(results)}] Saved visualization to: {output_file}")
            except Exception as exc:
                print(f"[{result_number}/{len(results)}] Visualization skipped: {exc}")


if __name__ == "__main__":
    main()
