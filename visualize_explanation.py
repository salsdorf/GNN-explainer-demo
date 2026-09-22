import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw

from explanation_runner import CLASS_MEANINGS, visualization_path

"""
Visualizer for imporantance values in MUTAG molecule Graphs. 
Outputs a png of the molecule where edges and nodes are colorized according to their explanation values.
Importance goes from blue to red (low-high)
"""

ATOM_NUMBERS = {
    0: 6,
    1: 7,
    2: 8,
    3: 9,
    4: 53,
    5: 17,
    6: 35,
}

BOND_TYPES = {
    0: "aromatic",
    1: "single",
    2: "double",
    3: "triple",
}

IMPORTANCE_CMAP = LinearSegmentedColormap.from_list(
    "importance_blue_yellow_red",
    ["#3f8fc4", "#ffe082", "#d94f5c"],
)
IMPORTANCE_NORM = Normalize(vmin=0.0, vmax=1.0)


def _build_molecule(data):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for visualization. "
            "Select the RDKit-enabled Python environment."
        ) from exc

    molecule = Chem.RWMol()
    for node_index, feature_index in enumerate(data.x.argmax(dim=-1).tolist()):
        atom = Chem.Atom(ATOM_NUMBERS[int(feature_index)])
        atom.SetProp("atomLabel", atom.GetSymbol())
        atom.SetProp("atomNote", str(node_index))
        molecule.AddAtom(atom)

    edge_lookup = {}
    bond_names = {}
    for edge_index in range(data.edge_index.size(1)):
        source, target = data.edge_index[:, edge_index].tolist()
        edge = tuple(sorted((int(source), int(target))))
        if edge in edge_lookup:
            continue

        bond_index = int(data.edge_attr[edge_index].argmax().item())
        edge_lookup[edge] = edge_index
        bond_name = BOND_TYPES.get(bond_index, "single")
        bond_names[edge] = bond_name
        bond_type = {
            "aromatic": Chem.BondType.AROMATIC,
            "single": Chem.BondType.SINGLE,
            "double": Chem.BondType.DOUBLE,
            "triple": Chem.BondType.TRIPLE,
        }[bond_name]
        molecule.AddBond(edge[0], edge[1], bond_type)

    molecule = molecule.GetMol()
    for bond in molecule.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
            molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsAromatic(True)
            molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsAromatic(True)

    rdDepictor.Compute2DCoords(molecule)
    return molecule, edge_lookup, bond_names


def _importance_color(value):
    return tuple(IMPORTANCE_CMAP(IMPORTANCE_NORM(float(value)))[:3])


def save_visualization(result, output_root="output"):
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError as exc:
        raise RuntimeError(
            "RDKit drawing support is required for visualization."
        ) from exc

    data = result["data"]
    explanation = result["explanation"]
    molecule, edge_lookup, bond_names = _build_molecule(data)

    node_mask = explanation.get("node_mask")
    edge_values = explanation.edge_mask.detach().cpu().tolist()

    atom_colors = {}
    if node_mask is not None:
        node_values = node_mask.abs().mean(dim=-1).detach().cpu().tolist()
        atom_colors = {
            node_index: _importance_color(value)
            for node_index, value in enumerate(node_values)
        }
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
    options = drawer.drawOptions()
    options.atomHighlightsAreCircles = True
    options.fillHighlights = True
    options.splitBonds = False
    options.bondLineWidth = 0
    options.highlightBondWidthMultiplier = 1
    options.fixedFontSize = 16
    options.annotationFontScale = 0.75
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=list(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
    )
    drawer.FinishDrawing()

    molecule_image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    image_draw = ImageDraw.Draw(molecule_image)
    node_radius = 28
    for bond in molecule.GetBonds():
        edge = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        source = drawer.GetDrawCoords(edge[0])
        target = drawer.GetDrawCoords(edge[1])
        dx = target.x - source.x
        dy = target.y - source.y
        length = (dx * dx + dy * dy) ** 0.5
        unit_x = dx / length
        unit_y = dy / length
        perpendicular_x = -unit_y
        perpendicular_y = unit_x
        start = (source.x + unit_x * node_radius, source.y + unit_y * node_radius)
        end = (target.x - unit_x * node_radius, target.y - unit_y * node_radius)
        bond_name = bond_names[edge]
        line_count = {"single": 1, "aromatic": 1, "double": 2, "triple": 3}[bond_name]
        offsets = {1: [0], 2: [-5, 5], 3: [-7, 0, 7]}[line_count]
        color = tuple(int(channel * 255) for channel in _importance_color(edge_values[edge_lookup[edge]]))
        line_width = 6 if bond_name == "aromatic" else 3
        for offset in offsets:
            offset_x = perpendicular_x * offset
            offset_y = perpendicular_y * offset
            image_draw.line(
                [(start[0] + offset_x, start[1] + offset_y), (end[0] + offset_x, end[1] + offset_y)],
                fill=color,
                width=line_width,
            )

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(molecule_image)
    axis.axis("off")

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=IMPORTANCE_CMAP,
        norm=IMPORTANCE_NORM,
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, ax=axis, fraction=0.045, pad=0.1)
    colorbar.ax.set_title("Importance", pad=8)

    true_class = int(data.y.item())
    predicted_class = result["predicted_class"]
    explanation_type = result["explanation_type"]
    axis.set_title(
        f"MUTAG graph explanation ({explanation_type})\n"
        f"Predicted class: {predicted_class} "
        f"({CLASS_MEANINGS.get(predicted_class, 'unknown')})"
    )

    output_file = visualization_path(
        result["index"],
        true_class,
        explanation_type,
        output_root=output_root,
    )
    figure.tight_layout()
    figure.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return output_file
