import torch
import torch.nn as nn


class AntonModel(nn.Module):
    def __init__(self, initial_vertex_positions: torch.Tensor):
        super().__init__()
        self.initial_vertex_positions = initial_vertex_positions
        # todo, change it back
        self.weights = nn.Parameter(torch.zeros_like(initial_vertex_positions), requires_grad=True)  # [vertex, xyz]

    def forward(self, face_vertices: torch.Tensor):
        # face_vertices: [face_id, vertex_id] padded with -1
        dist = self.weights - self.initial_vertex_positions  # [vertex, xyz]
        closeness = torch.einsum("ij,ik->", dist, dist)

        planarness = 0

        return closeness + planarness


def main():
    initial_vertex_positions = torch.rand([30, 3])
    face_vertices = torch.randint(0, 30, (20, 4))
    model = AntonModel(initial_vertex_positions=initial_vertex_positions)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    while True:
        optimizer.zero_grad()

        # Make predictions for this batch
        loss = model(face_vertices)

        # Compute gradient
        loss.backward()

        optimizer.step()

        print(loss.item())

        if loss <= 0.000001:
            break

main()