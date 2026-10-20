"""Forward and inverse kinematics of the FR5 model, read from the GLB the 3D view shows.

The chain is whatever the file says: the rest transform of every node from the scene
root down to ``tcp``, and for the joint nodes the axis and limits of their row in
``extras.arkitekt.joints``. A joint turns its node as the viewer does, about the node's
origin and an axis in its own frame (``q_rest * axisAngle(axis, angle)``), so a pose
computed here is the pose drawn there.

Inverse kinematics is numeric: damped least squares on the TCP's position and rotation
error, from a few starting poses, keeping the solution that moves the arm least.
"""

import json
import struct
from typing import Any

import numpy as np

import fairino_model

#: When a solution counts as there: metres, and radians of rotation.
POSITION_TOLERANCE = 1e-3
ROTATION_TOLERANCE = np.deg2rad(0.5)
#: How much a radian of rotation error weighs against a metre of position error.
ROTATION_WEIGHT = 0.3
DAMPING = 0.05
ITERATIONS = 300
#: Starting poses tried after the current and the folded one, the same every time.
RANDOM_SEEDS = 8


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _log(r: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix (axis times angle)."""
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.pi - angle < 1e-6:
        # A half turn: the axis is the eigenvector of eigenvalue one.
        values, vectors = np.linalg.eigh((r + r.T) / 2)
        return angle * vectors[:, int(np.argmax(values))]
    return angle / (2 * np.sin(angle)) * vector


class Chain:
    """The nodes from the scene root to the TCP, the joints among them, and their limits."""

    def __init__(self, rests: list[np.ndarray], joints: list[tuple[int, str, np.ndarray, float, float]]) -> None:
        self.rests = rests
        #: ``(index into rests, name, axis, lower, upper)`` per joint, base first.
        self.joints = joints
        self.names = tuple(name for _, name, _, _, _ in joints)
        self.lower = np.array([lower for *_, lower, _ in joints])
        self.upper = np.array([upper for *_, upper in joints])

    @classmethod
    def from_glb(cls, data: bytes, tip: str = "tcp") -> "Chain":
        json_length = struct.unpack_from("<I", data, 12)[0]
        document: dict[str, Any] = json.loads(data[20 : 20 + json_length])
        nodes = document["nodes"]
        parents = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
        index = next(i for i, node in enumerate(nodes) if node.get("name") == tip)
        path = [index]
        while path[-1] in parents:
            path.append(parents[path[-1]])
        path.reverse()

        table = {row["node"]: row for row in document["extras"]["arkitekt"]["joints"]}
        joints = []
        for position, node in enumerate(path):
            row = table.get(nodes[node].get("name"))
            if row is None:
                continue
            if row["type"] != "revolute":
                raise ValueError(f"joint {row['name']!r} is {row['type']}: only revolute joints are solved")
            axis = np.array(row["axis"], dtype=float)
            joints.append((position, row["name"], axis / np.linalg.norm(axis), row["limit"]["lower"], row["limit"]["upper"]))
        return cls([fairino_model._matrix(nodes[node]) for node in path], joints)

    def forward(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The TCP's 4x4 in the scene's frame, and each joint's origin and axis there."""
        turns = {position: (axis, angle) for (position, _, axis, _, _), angle in zip(self.joints, angles)}
        placed = np.eye(4)
        origins, axes = [], []
        for position, rest in enumerate(self.rests):
            placed = placed @ rest
            if position in turns:
                axis, angle = turns[position]
                origins.append(placed[:3, 3].copy())
                axes.append(placed[:3, :3] @ axis)
                turn = np.eye(4)
                turn[:3, :3] = _axis_angle(axis, angle)
                placed = placed @ turn
        return placed, np.array(origins), np.array(axes)

    def solve(self, position: np.ndarray, rotation: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """Joint angles putting the TCP at ``position`` turned to ``rotation``, from ``seed``; ``None`` if it gets stuck."""
        angles = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        for _ in range(ITERATIONS):
            tcp, origins, axes = self.forward(angles)
            position_error = position - tcp[:3, 3]
            rotation_error = _log(rotation @ tcp[:3, :3].T)
            if np.linalg.norm(position_error) < POSITION_TOLERANCE and np.linalg.norm(rotation_error) < ROTATION_TOLERANCE:
                return angles
            jacobian = np.vstack([np.cross(axes, tcp[:3, 3] - origins).T, ROTATION_WEIGHT * axes.T])
            error = np.concatenate([position_error, ROTATION_WEIGHT * rotation_error])
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + DAMPING**2 * np.eye(6), error)
            angles = np.clip((angles + step + np.pi) % (2 * np.pi) - np.pi, self.lower, self.upper)
        return None

    def reach(self, position: np.ndarray, current: np.ndarray, floor: float = fairino_model.FLOOR_Y) -> np.ndarray:
        """Joint angles putting the TCP at ``position`` without turning it from where ``current`` has it.

        Of the solutions that keep every joint and the TCP above ``floor``, the one whose
        largest joint change from ``current`` is smallest. Raises ``ValueError`` if none is found.
        """
        position = np.asarray(position, dtype=float)
        current = np.asarray(current, dtype=float)
        rotation = self.forward(current)[0][:3, :3]
        random = np.random.default_rng(0)
        seeds = [current, np.zeros(len(self.joints))]
        seeds += [random.uniform(self.lower, self.upper) for _ in range(RANDOM_SEEDS)]
        best, best_change = None, np.inf
        for seed in seeds:
            angles = self.solve(position, rotation, seed)
            if angles is None:
                continue
            tcp, origins, _ = self.forward(angles)
            if origins[1:, 1].min() < floor or tcp[1, 3] < floor:
                continue
            change = float(np.abs(angles - current).max())
            if change < best_change:
                best, best_change = angles, change
        if best is None:
            x, y, z = position
            raise ValueError(f"can't reach ({x:.3f}, {y:.3f}, {z:.3f}) with the flange as it is now")
        return best
