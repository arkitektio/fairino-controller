"""Offline checks of the FR5 kinematics and the two GLBs the app shows."""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import fairino_kinematics
import fairino_model

VALIDATION = Path.home() / "Code/deployments/next/mounts/rekuest/facade/glb_validation.py"


@pytest.fixture(scope="module")
def arm() -> bytes:
    return fairino_model.build()


@pytest.fixture(scope="module")
def chain(arm: bytes) -> fairino_kinematics.Chain:
    return fairino_kinematics.Chain.from_glb(arm)


def _document(data: bytes) -> dict:
    return json.loads(data[20 : 20 + struct.unpack_from("<I", data, 12)[0]])


def test_tcp_is_the_flange_face(chain: fairino_kinematics.Chain) -> None:
    export = fairino_model._Export(fairino_model.SOURCE.read_bytes())
    joints, _, tcp = fairino_model._kinematics(export)
    # The assembly hangs in the scene turned a quarter of a half turn about Y.
    turned = fairino_kinematics._axis_angle(np.array([0.0, 1.0, 0.0]), -np.pi / 4)
    assert np.allclose(chain.forward(np.zeros(6))[0][:3, 3], turned @ tcp, atol=1e-6)
    # The face is 0.0984 m out from the last pivot (measured on the export).
    assert np.linalg.norm(tcp - joints[5][0]) == pytest.approx(0.0984, abs=1e-3)


def test_round_trip(chain: fairino_kinematics.Chain) -> None:
    random = np.random.default_rng(1)
    checked = solved = 0
    while checked < 50:
        pose = random.uniform(chain.lower, chain.upper)
        tcp, origins, _ = chain.forward(pose)
        if origins[1:, 1].min() < fairino_model.FLOOR_Y or tcp[1, 3] < fairino_model.FLOOR_Y:
            continue
        checked += 1
        # From a nearby pose, back to where ``pose`` puts the flange, turned as it turns it.
        start = np.clip(pose + random.uniform(-0.5, 0.5, 6), chain.lower, chain.upper)
        angles = chain.solve(tcp[:3, 3], tcp[:3, :3], start)
        if angles is None:
            continue
        solved += 1
        reached = chain.forward(angles)[0]
        assert np.linalg.norm(reached[:3, 3] - tcp[:3, 3]) < 1e-3
        assert np.linalg.norm(fairino_kinematics._log(reached[:3, :3] @ tcp[:3, :3].T)) < np.deg2rad(0.5)
    # 199 of 200 solved when this was written.
    assert solved >= 47, f"{solved}/{checked}"


def test_reach_keeps_the_flange_turned(chain: fairino_kinematics.Chain) -> None:
    current = np.deg2rad([0, -60, 100, 60, 0, 0])
    target = np.array([0.4, 0.3, 0.2])
    angles = chain.reach(target, current)
    before, after = chain.forward(current)[0], chain.forward(angles)[0]
    assert np.linalg.norm(after[:3, 3] - target) < 1e-3
    assert np.linalg.norm(fairino_kinematics._log(after[:3, :3] @ before[:3, :3].T)) < np.deg2rad(0.5)


def test_most_of_the_floor_is_reachable(chain: fairino_kinematics.Chain) -> None:
    points = [
        (x, y, z)
        for y in (0.06, 0.15, 0.30, 0.45)
        for x in np.linspace(-0.8, 0.8, 9)
        for z in np.linspace(-0.8, 0.8, 9)
        if 0.25 < np.hypot(x, z) < 0.85
    ]
    reached = 0
    for point in points:
        try:
            chain.reach(np.array(point), np.zeros(6))
            reached += 1
        except ValueError:
            pass
    assert reached >= 0.8 * len(points), f"{reached}/{len(points)}"


@pytest.mark.parametrize("name", list(fairino_model.PEDESTALS))
def test_pedestal_tops_are_reachable_folded(chain: fairino_kinematics.Chain, name: str) -> None:
    x, z, height = fairino_model.PEDESTALS[name]
    chain.reach(np.array([x, fairino_model.FLOOR_Y + height + 0.08, z]), np.zeros(6))


def test_out_of_reach_raises(chain: fairino_kinematics.Chain) -> None:
    with pytest.raises(ValueError, match="can't reach"):
        chain.reach(np.array([2.0, 0.3, 0.0]), np.zeros(6))


@pytest.mark.parametrize("build, path", [(fairino_model.build, fairino_model.ARM), (fairino_model.workspace, fairino_model.WORKSPACE)])
def test_the_models_are_up_to_date(build, path: Path) -> None:
    assert path.read_bytes() == build(), f"run `python fairino_model.py` to write {path.name} again"


def test_arm_keeps_its_joint_table(arm: bytes) -> None:
    document = _document(arm)
    names = {node.get("name") for node in document["nodes"]}
    assert "tcp" in names
    assert [row["name"] for row in document["extras"]["arkitekt"]["joints"]] == list(fairino_model.JOINTS)


def test_workspace_is_one_clickable_node() -> None:
    document = _document(fairino_model.workspace())
    floor = next(node for node in document["nodes"] if node.get("name") == "floor")
    children = {document["nodes"][child]["name"] for child in floor["children"]}
    assert children == {"floor_slab", *fairino_model.PEDESTALS}
    assert document["scenes"][0]["nodes"] == [document["nodes"].index(floor)]


@pytest.mark.skipif(not VALIDATION.exists(), reason="the rekuest server mount is not here")
@pytest.mark.parametrize("build", [fairino_model.build, fairino_model.workspace], ids=["arm", "workspace"])
def test_the_server_accepts_it(build) -> None:
    spec = importlib.util.spec_from_file_location("glb_validation", VALIDATION)
    validation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validation  # its dataclasses look their module up
    spec.loader.exec_module(validation)
    data = build()
    report = validation.validate_model(data, validation.model_ref(data))
    assert not [warning for warning in report.warnings if "node names" in warning]
