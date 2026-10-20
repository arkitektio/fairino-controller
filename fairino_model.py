"""The Fairino FR5 as a model a Robot blok can drive: ``models/fairino_cad.glb`` given joints.

The file is a CAD export. Its seven bodies (base, shoulder, upper arm, forearm, three
wrist bodies) hang side by side under one node, so nothing in it says which body turns
on which, or about what. This module reads that off the geometry and writes a GLB
whose bodies hang on six pivot nodes, one inside the next, with the joint table the
viewer applies (``extras.arkitekt.joints``):

* every joint module is closed by a round cover held by four screws, so the centre of
  a cover's screws lies on a joint axis and their plane is square to it;
* the two wrist axes meet where the forearm's and the first wrist body's axes cross.

A seventh node, ``tcp``, hangs under the last joint at the centre of the flange's outer
face: the point inverse kinematics puts where it is asked to (``fairino_kinematics``).

One thing in the export is wrong and is put right here: the second wrist body and the
tool flange sit *inside* the first wrist body, on the same axis, instead of on its
output face. They are moved there, with the flange axis square to the fifth joint as
it is on the arm.

Joint values are radians from the pose the export is in (the arm folded), which is
not the controller's zero.
"""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MODELS = Path(__file__).parent / "models"
SOURCE = MODELS / "fairino_cad.glb"
#: What ``build`` and ``workspace`` write: the files the app registers as its assets.
ARM = MODELS / "fairino_robot.glb"
WORKSPACE = MODELS / "fairino_workspace.glb"

#: The node the seven bodies hang under, and their names in kinematic order.
ASSEMBLY = "FR5-V60_Model1"
BODIES = ("base", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6")
#: The scale the export gives a cover screw: how the screws are told from the rest.
SCREW_SCALE = 0.004825
#: How far a joint turns either way from the exported pose, in radians.
TRAVEL = np.deg2rad(175.0)
#: The height of the base's underside: the top of the workspace floor.
FLOOR_Y = -0.0101


def _rotation(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion(r: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        x, y, z = (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)
    else:
        # A half turn: the axis is the eigenvector of eigenvalue one.
        values, vectors = np.linalg.eigh((r + r.T) / 2)
        x, y, z = vectors[:, int(np.argmax(values))]
    q = np.array([x, y, z, w])
    return [float(v) for v in q / np.linalg.norm(q)]


def _matrix(node: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rotation(node.get("rotation", [0, 0, 0, 1])) @ np.diag(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def _closest_points(p: np.ndarray, u: np.ndarray, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the lines ``p + s u`` and ``q + t v`` come closest (``u``, ``v`` unit)."""
    w, b = p - q, float(u @ v)
    s = (b * (v @ w) - (u @ w)) / (1 - b * b)
    t = ((v @ w) - b * (u @ w)) / (1 - b * b)
    return p + s * u, q + t * v


class _Export:
    """The exported file: its JSON, its binary chunk, and the bodies in the assembly's frame."""

    def __init__(self, data: bytes) -> None:
        json_length = struct.unpack_from("<I", data, 12)[0]
        self.document: dict[str, Any] = json.loads(data[20 : 20 + json_length])
        self.rest = data[20 + json_length :]
        self.binary = self.rest[8:]
        self.nodes: list[dict[str, Any]] = self.document["nodes"]
        self.assembly = next(i for i, node in enumerate(self.nodes) if node.get("name") == ASSEMBLY)
        self.bodies: list[int] = list(self.nodes[self.assembly]["children"])
        if len(self.bodies) != len(BODIES):
            raise ValueError(f"expected {len(BODIES)} bodies under {ASSEMBLY}, found {len(self.bodies)}")

    def _positions(self, mesh: int) -> np.ndarray:
        chunks = []
        for primitive in self.document["meshes"][mesh]["primitives"]:
            accessor = self.document["accessors"][primitive["attributes"]["POSITION"]]
            view = self.document["bufferViews"][accessor["bufferView"]]
            offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            stride = view.get("byteStride", 12)
            raw = np.frombuffer(self.binary, dtype=np.uint8, count=(accessor["count"] - 1) * stride + 12, offset=offset)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(accessor["count"], 12), strides=(stride, 1))
            chunks.append(rows.copy().view("<f4").reshape(-1, 3))
        return np.concatenate(chunks)

    def solids(self, body: int, *, screws: bool) -> list[np.ndarray]:
        """The vertices of each solid of a body, in the assembly's frame: its screws, or everything else."""
        placed = _matrix(self.nodes[self.bodies[body]])
        found = []
        for child in self.nodes[self.bodies[body]].get("children", []):
            node = self.nodes[child]
            if (abs(node.get("scale", [1])[0] - SCREW_SCALE) < 1e-4) != screws:
                continue
            world = placed @ _matrix(node)
            found.append(self._positions(node["mesh"]) @ world[:3, :3].T + world[:3, 3])
        return found

    def covers(self, body: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each screwed cover of a body: the centre of its four screws and the normal of their plane."""
        centres = np.array([solid.mean(0) for solid in self.solids(body, screws=True)])
        covers, left = [], list(range(len(centres)))
        while left:
            nearest = sorted(left, key=lambda i: float(np.linalg.norm(centres[i] - centres[left[0]])))[:4]
            left = [i for i in left if i not in nearest]
            ring = centres[nearest]
            centre = ring.mean(0)
            covers.append((centre, np.linalg.svd(ring - centre)[2][2]))
        return covers


def _kinematics(export: _Export) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """The six joints as ``(pivot, axis)`` in the assembly's frame, where the last two bodies belong, and the TCP."""
    up = np.array([0.0, 1.0, 0.0])

    lower, upper = sorted(export.covers(2), key=lambda cover: cover[0][1])
    across = lower[1] * np.sign(lower[1][0])  # the shoulder, elbow and first wrist axis
    shoulder = np.array([0.0, lower[0][1], 0.0])  # where that axis crosses the vertical one
    elbow = upper[0]

    (forearm_cover,) = export.covers(3)
    (wrist_cover,) = export.covers(4)
    along = wrist_cover[1] * np.sign(wrist_cover[1][1])  # the second wrist axis
    wrist, _ = _closest_points(forearm_cover[0], across, wrist_cover[0], along)

    # The first wrist body ends, away from its cover, in the face the next body mounts on.
    first = np.concatenate(export.solids(4, screws=False)) - wrist
    face = -float((first @ along).min())
    # The second wrist body has one port sticking out sideways: that is what mounts.
    second = np.concatenate(export.solids(5, screws=False)) - wrist
    radial = second - np.outer(second @ along, along)
    reach = np.linalg.norm(radial, axis=1)
    port = radial[reach > 0.85 * reach.max()].mean(0)
    port /= np.linalg.norm(port)
    mounted = wrist - (face + float((radial @ port).max())) * along

    # Its own axis, which the flange turns on, goes from along the first wrist body to
    # across it; its port from sideways to facing the mounting face.
    source = np.stack([along, port, np.cross(along, port)], axis=1)
    target = np.stack([across, along, np.cross(across, along)], axis=1)
    repair = np.eye(4)
    repair[:3, :3] = target @ source.T
    repair[:3, 3] = mounted - repair[:3, :3] @ wrist

    # The flange, where it is mounted, ends in its outer face: the TCP is that face's centre.
    flange = np.concatenate(export.solids(6, screws=False)) @ repair[:3, :3].T + repair[:3, 3]
    out = (flange - mounted) @ across
    tcp = mounted + float(out[np.argmax(np.abs(out))]) * across

    joints = [(np.zeros(3), up), (shoulder, across), (elbow, across), (wrist, across), (wrist, along), (mounted, across)]
    return joints, repair, tcp


def build(source: Path = SOURCE) -> bytes:
    """The FR5 with six pivot nodes and a joint table, as GLB bytes."""
    export = _Export(source.read_bytes())
    joints, repair, tcp = _kinematics(export)
    nodes = export.nodes

    taken: set[str] = set()
    for node in nodes:
        # Lowercase names, as blok bindings address nodes.
        name = "".join(c if c.isalnum() else "_" for c in node.get("name", "node").lower())
        while name in taken:
            name += "_"
        taken.add(name)
        node["name"] = name

    parent = export.assembly
    nodes[parent]["children"] = [export.bodies[0]]
    nodes[export.bodies[0]]["name"] = BODIES[0]
    table = []
    previous = np.zeros(3)
    for index, (pivot, axis) in enumerate(joints, start=1):
        name = f"joint_{index}"
        body = nodes[export.bodies[index]]
        placed = _matrix(body)
        if index >= 5:
            placed = repair @ placed
        body["name"] = BODIES[index]
        body["rotation"] = _quaternion(placed[:3, :3])
        body["translation"] = [float(v) for v in placed[:3, 3] - pivot]
        nodes.append({"name": name, "translation": [float(v) for v in pivot - previous], "children": [export.bodies[index]]})
        nodes[parent].setdefault("children", []).append(len(nodes) - 1)
        parent, previous = len(nodes) - 1, pivot
        table.append(
            {
                "name": name,
                "node": name,
                "type": "revolute",
                "axis": [round(float(v), 6) for v in axis / np.linalg.norm(axis)],
                "limit": {"lower": -float(TRAVEL), "upper": float(TRAVEL)},
            }
        )

    nodes.append({"name": "tcp", "translation": [float(v) for v in tcp - previous]})
    nodes[parent].setdefault("children", []).append(len(nodes) - 1)

    export.document["extras"] = {"arkitekt": {"version": 1, "joints": table}}
    return _glb(export.document, export.rest)


def _glb(document: dict[str, Any], rest: bytes) -> bytes:
    """A GLB of a document and what follows its JSON chunk (the binary chunk, header included)."""
    payload = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    body_bytes = struct.pack("<II", len(payload), 0x4E4F534A) + payload + rest
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body_bytes)) + body_bytes


#: Boxes the arm can be sent to, by name: ``(x, z)`` of the centre and the height of the top above the floor, metres.
PEDESTALS = {"pedestal_1": (0.45, 0.25, 0.10), "pedestal_2": (-0.15, 0.5, 0.25), "pedestal_3": (0.35, -0.4, 0.40)}
#: The pedestals' side, and the floor's.
PEDESTAL_SIDE = 0.12
FLOOR_SIDE = 2.0


def _cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit cube centred on the origin: positions and normals per face corner, and triangle indices."""
    positions, normals, indices = [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
            base = len(positions)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                positions.append(0.5 * (normal + a * u + b * v))
                normals.append(normal)
            # Counter-clockwise seen from outside, whichever way the face looks.
            quad = (0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)
            indices.extend(base + i for i in quad)
    return np.array(positions, "<f4"), np.array(normals, "<f4"), np.array(indices, "<u2")


def workspace() -> bytes:
    """The floor the arm stands on, with pedestals of a few heights, as GLB bytes.

    One node, ``floor``, holds every box, so a click anywhere on it reaches one handler.
    Its top is the base's underside (``FLOOR_Y``) and the scene's frame is the arm's.
    """
    positions, normals, indices = _cube()
    binary = positions.tobytes() + normals.tobytes() + indices.tobytes()
    binary += b"\0" * (-len(binary) % 4)
    nodes: list[dict[str, Any]] = [{"name": "floor", "children": []}]

    def box(name: str, centre: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        nodes.append({"name": name, "mesh": 0, "translation": list(centre), "scale": list(size)})
        nodes[0]["children"].append(len(nodes) - 1)

    thickness = 0.02
    box("floor_slab", (0.0, FLOOR_Y - thickness / 2, 0.0), (FLOOR_SIDE, thickness, FLOOR_SIDE))
    for name, (x, z, height) in PEDESTALS.items():
        box(name, (x, FLOOR_Y + height / 2, z), (PEDESTAL_SIDE, height, PEDESTAL_SIDE))

    document = {
        "asset": {"version": "2.0", "generator": "fairino_model.workspace"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"name": "box", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.57, 0.6, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": [-0.5] * 3, "max": [0.5] * 3},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": normals.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": positions.nbytes + normals.nbytes, "byteLength": indices.nbytes, "target": 34963},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    return _glb(document, struct.pack("<II", len(binary), 0x004E4942) + binary)


JOINTS = tuple(f"joint_{index}" for index in range(1, 7))

if __name__ == "__main__":
    for target, data in ((ARM, build()), (WORKSPACE, workspace())):
        target.write_bytes(data)
        print(f"wrote {target}")
