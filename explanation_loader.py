# ignore this file, it is a work in progress

import json
from pathlib import Path


def load_explanations() -> dict:
	"""Load explanation data from explanations.json."""
	path = Path(__file__).with_name("explanations.json")
	with path.open(encoding="utf-8") as file:
		return json.load(file)

def top_node_edge_involvement(explanation:dict): 
	# for a given explanation, returns the ratio of top nodes that are also involved in top edges
	# as well as a list of the ones that are 
	top_nodes_list = explanation.get("top-nodes", [])
	top_edges_list = explanation.get("top-edges", [])
	nodes_in_top_edges_and_nodes = set()

	for edge in top_edges_list:
		unique_involved_atoms = set()
		unique_involved_atoms.add(edge["source"])
		unique_involved_atoms.add(edge["target"])
		for node in top_nodes_list:
			if edge["source"] == node["node_index"]:
				nodes_in_top_edges_and_nodes.add(node["node_index"])
			if edge["target"] == node["node_index"]:
				nodes_in_top_edges_and_nodes.add(node["node_index"])

	node_involvement_ratio = len(unique_involved_atoms)/len(nodes_in_top_edges_and_nodes)
	return node_involvement_ratio, nodes_in_top_edges_and_nodes

def compare_explanation(ex1, ex2):
	"""
	Compare two given explanations on the same graph. 
	Returns the ratio of shared nodes and edges, as well as the actual shared nodes and edges with their importances.
	"""

	involved_nodes1 = ex1.get("top-nodes", [])
	involved_edges1 = ex1.get("top-edges", [])

	# this expects the json to be formed uniformly
	# nr_nodes = len(involved_nodes1)
	# nr_edges =len(involved_edges1)

	involved_nodes2 = ex2.get("top-nodes", [])
	involved_edges2 = ex2.get("top-edges", [])

	shared_nodes = set()
	shared_edges = set()

	unique_edges = set()
	unique_nodes = set()

	for e in involved_edges1:
		unique_edges.add((e["source"], e["target"]))
		for e2 in involved_edges2:
			unique_edges.add((e2["source"], e2["target"]))
			if e["target"] == e2["target"] and e["source"] == e2["source"]:	
				shared_edges.add(((e["source"], e["target"]), (e["importance"], e2["importance"])))

	for n in involved_nodes1:
		unique_nodes.add(n["node_index"])
		for n2 in involved_nodes2:
			unique_nodes.add(n2["node_index"])
			if n["node_index"] == n2["node_index"]:	
				shared_nodes.add((n["node_index"], (n["importance"], n2["importance"])))

	shared_node_ratio = len(shared_nodes)/len(unique_nodes) if len(unique_nodes) > 0 else 0
	shared_edge_ratio = len(shared_edges)/len(unique_edges) if len(unique_edges) > 0 else 0

	return {
		"shared_node_ratio": shared_node_ratio,
		"shared_edge_ratio": shared_edge_ratio,
		"shared_nodes": shared_nodes, # nested tuple of node_idx and importances
		"shared_edges": shared_edges, # nested tuple of (source, target) and importances
	}


explanations = load_explanations()
