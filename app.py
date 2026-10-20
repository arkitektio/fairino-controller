"""A Fairino FR5 shown in 3D that follows the agent's joint state.

Run:  python app.py   (or `arkitekt run dev`)

The arm is simulated: its actions move six numbers, not a robot. What it shows is the
real CAD model. ``models/fairino_cad.glb`` has no joints, so ``fairino_model`` gives it
six and writes ``models/fairino_robot.glb``; the app registers that file as the asset
``arm`` and its scene blok names it as ``@self.assets.arm``. When the app runs it
uploads the file if the organization has not stored it yet. After changing
``fairino_model``, run ``python fairino_model.py`` to write the models again.

Joint positions are radians from the pose the CAD model is in (the arm folded), which
is not the controller's zero.

Click the floor or a pedestal in the 3D view and the arm goes there: ``move_to`` works
out the six joint angles that put the flange's face just above the clicked point, turned
as it is now (``fairino_kinematics``), and moves. The floor is a second asset,
``workspace``, because the view only reports where a click lands on a model.
"""

import math
from typing import Annotated

import koil
from arkitekt import App, Description, bsx, run

import fairino_kinematics
import fairino_model

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.1.0"

JOINTS = fairino_model.JOINTS
SPEED = 1.0  # rad/s, every joint
TRAVEL = math.radians(175.0)
#: The flange points sideways, so its rim hangs ~5 cm below its centre: this clears it by ~3 cm.
HOVER = 0.08
CHAIN = fairino_kinematics.Chain.from_glb(fairino_model.ARM.read_bytes())

app = App(
    "fairino-controller",
    __version__,
    author="jhnnsrs",
    description="A simulated Fairino FR5 with a live 3D view of its six joints: click the floor to send it there.",
)
app.register_asset("arm", "models/fairino_robot.glb")
app.register_asset("workspace", "models/fairino_workspace.glb")


@app.state(name="arm")
class ArmState:
    """Where the arm's joints are (``joints``) and where they are going (``targets``).

    Radians from the model's folded pose, keyed by joint name (``joint_1`` at the base
    to ``joint_6`` at the flange): the 3D view applies the maps to the model's joint
    table as they are.
    """

    joints: dict[str, float]
    targets: dict[str, float]
    speeds: dict[str, float]
    moving: bool = False


@app.startup
def start() -> ArmState:
    folded = dict.fromkeys(JOINTS, 0.0)
    return ArmState(joints=dict(folded), targets=dict(folded), speeds=dict.fromkeys(JOINTS, SPEED))


def travel(state: ArmState, targets: dict[str, float]) -> None:
    """Move joints together: publish the targets once, then the positions as they get there.

    Each joint runs at its own speed and stops when it arrives, as the view assumes: it
    glides to the targets at the published speeds and is corrected by each position.
    """
    targets = {name: min(max(position, -TRAVEL), TRAVEL) for name, position in targets.items()}
    begun = dict(state.joints)
    durations = {name: abs(target - begun[name]) / state.speeds[name] for name, target in targets.items()}
    longest = max(durations.values(), default=0.0)
    state.targets = {**state.targets, **targets}
    state.moving = True
    steps = max(1, int(longest / 0.25))
    for step in range(1, steps + 1):
        koil.sleep(longest / steps)
        elapsed = longest * step / steps
        state.joints = {
            **state.joints,
            **{
                name: target if elapsed >= durations[name] else begun[name] + (target - begun[name]) * elapsed / durations[name]
                for name, target in targets.items()
            },
        }
    state.moving = False


@app.action
def move_joints(
    state: ArmState,
    joint_1: Annotated[float, Description("Base, in degrees from the folded pose")] = 0.0,
    joint_2: Annotated[float, Description("Shoulder, in degrees from the folded pose")] = 0.0,
    joint_3: Annotated[float, Description("Elbow, in degrees from the folded pose")] = 0.0,
    joint_4: Annotated[float, Description("First wrist joint, in degrees from the folded pose")] = 0.0,
    joint_5: Annotated[float, Description("Second wrist joint, in degrees from the folded pose")] = 0.0,
    joint_6: Annotated[float, Description("Flange, in degrees from the folded pose")] = 0.0,
) -> None:
    """Move Joints

    Moves all six joints to the given angles, together.
    """
    degrees = (joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
    travel(state, {name: math.radians(angle) for name, angle in zip(JOINTS, degrees)})


@app.action
def fold(state: ArmState) -> None:
    """Fold

    Sends the arm back to the folded pose its model was drawn in.
    """
    travel(state, dict.fromkeys(JOINTS, 0.0))


@app.action
def reach_out(state: ArmState) -> None:
    """Reach Out

    Unfolds the arm: the upper arm leans back and the forearm swings out over it.
    """
    travel(state, {"joint_2": math.radians(-60.0), "joint_3": math.radians(100.0), "joint_4": math.radians(60.0)})


@app.action
def look_around(state: ArmState) -> None:
    """Look Around

    Turns the base a quarter turn each way and comes back, wherever the arm is.
    """
    begun = state.joints["joint_1"]
    for offset in (math.pi / 2, -math.pi / 2, 0.0):
        travel(state, {"joint_1": begun + offset})


@app.action
def move_joint(state: ArmState, name: str, position: float) -> None:
    """Move Joint

    Moves one joint to a position in radians: what the 3D view calls when a joint is
    set by hand.
    """
    if name not in state.joints:
        raise ValueError(f"there is no joint {name!r}")
    travel(state, {name: position})


@app.action
def move_to(
    state: ArmState,
    point: Annotated[list[float], Description("Where to put the flange: x, y, z in metres, in the scene's frame")],
    hover: Annotated[float, Description("How far above the point the flange's centre stops, in metres")] = HOVER,
) -> list[float]:
    """Move To

    Moves the flange to a point, turned as it is now: what the 3D view calls when the
    floor is clicked. Returns the six joint angles it moved to, in degrees. The joints
    move together, so the flange is turned as before when it arrives, not on the way,
    and its path is not a straight line.
    """
    if len(point) != 3:
        raise ValueError(f"a point is x, y, z; got {len(point)} numbers")
    target = [point[0], point[1] + hover, point[2]]
    current = [state.joints[name] for name in CHAIN.names]
    angles = CHAIN.reach(target, current)
    travel(state, dict(zip(CHAIN.names, (float(angle) for angle in angles))))
    return [math.degrees(angle) for angle in angles]


app.blok(
    "fairino-panel",
    bsx("""
    <Card>
      <CardHeader><CardTitle text="Fairino FR5" /></CardHeader>
      <CardContent>
        <Row>
          <Button label="Reach out" onClick="@self.reach_out()" />
          <Button label="Look around" onClick="@self.look_around()" />
          <Button label="Fold" onClick="@self.fold()" />
        </Row>
      </CardContent>
    </Card>
    """),
    description="The arm's controls.",
    default=True,
)

app.blok(
    "fairino-3d",
    bsx("""
    <Scene camera="iso" height="100%">
      <Robot id="arm" src="@self.assets.arm"
             joints="@self.arm.joints"
             animateTo="@self.arm.targets"
             speed="@self.arm.speeds"
             onJointChange="@self.move_joint(name=event.name, position=event.value)">
        <OnClick node="base" onClick="@self.fold()" />
        <Popout node="link_6" trigger="hover">
          <Badge text="Tool flange" variant="outline" />
        </Popout>
      </Robot>
      <Model id="workspace" src="@self.assets.workspace">
        <OnClick node="floor" onClick="@self.move_to(point=event.point)" />
      </Model>
    </Scene>
    """),
    description="The arm in 3D: its six joints follow the agent, a joint set by hand moves the arm, and a click on the floor sends the flange there.",
    context="scene",
    default=True,
)


if __name__ == "__main__":
    run(app)
