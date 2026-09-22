"""
Script used to train a simple GIN model for running pyg evaluations on
Note that the aim was to simply generate a demo GNN checkpoint fit for evalution purposes
without keeping model quality in mind. 
Model was saved to gin_mutag.pt
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool


# Configuration

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
NUM_LAYERS = 5
EPOCHS = 100
LR = 0.001

print("Device:", DEVICE)


# class to build GNN

class GIN(torch.nn.Module):

    def __init__(
        self,
        num_features,
        hidden_channels,
        num_classes,
        num_layers,
    ):
        super().__init__()

        # initialize layers with empty list
        self.convs = torch.nn.ModuleList()

        # First GIN layer
        # a GIN layer consists of a neural network (MLP) and a sum aggregation function
        # the latter is implemented in GINConv
        nn = Sequential(
            Linear(num_features, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        self.convs.append(
            GINConv(nn)
        )

        # Remaining GIN layers
        for _ in range(num_layers - 1):

            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )

            self.convs.append(
                GINConv(nn)
            )

        # Graph-level classifier
        self.classifier = Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, x, edge_index, batch):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x) # sets negative values to zero 

        # Convert node embeddings -> graph embedding
        x = global_add_pool(x, batch)

        # Graph classification
        x = self.classifier(x)

        return x


def main():
    # Dataset
    dataset = TUDataset(
        root="data/MUTAG",
        name="MUTAG",
    )

    print("Number of graphs:", len(dataset))
    print("Number of node features:", dataset.num_features)
    print("Number of classes:", dataset.num_classes)

    # Train / test split
    torch.manual_seed(42)
    dataset = dataset.shuffle()

    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = GIN(
        num_features=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=dataset.num_classes,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            out = model(
                data.x,
                data.edge_index,
                data.batch,
            )

            loss = F.cross_entropy(
                out,
                data.y,
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(
                    data.x,
                    data.edge_index,
                    data.batch,
                )
                pred = out.argmax(dim=-1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

        accuracy = correct / total

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {total_loss / len(train_loader):.4f} | "
                f"Test accuracy {accuracy:.3f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_features": dataset.num_features,
                "hidden_channels": HIDDEN_CHANNELS,
                "num_classes": dataset.num_classes,
                "num_layers": NUM_LAYERS,
            },
            "dataset": "MUTAG",
        },
        "gin_mutag.pt",
    )

    print()
    print("Saved model to: gin_mutag.pt")


if __name__ == "__main__":
    main()
