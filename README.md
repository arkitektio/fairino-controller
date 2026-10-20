# fairino-controller

A simulated Fairino FR5 as an Arkitekt app. The actions move six joint numbers, not a
real robot. The 3D view (the `fairino-3d` scene blok) shows the real CAD model, which
follows those joints.

## Run

```sh
uv sync
uv run python app.py        # or: uv run arkitekt run dev
```

Open the `fairino-3d` blok and click the floor or one of the pedestals. `move_to`
works out the six joint angles (inverse kinematics) that put the flange 8 cm above
the clicked point, with the flange turned as it is now. It then moves the arm there
and returns the angles in degrees. If no joint angles reach the point, it raises
`can't reach …`. With the flange as it is in the folded pose, that is about 15% of
the floor, mostly the part close to the base.

## Layout

- `app.py` is the app: the arm state, its actions and two bloks (a panel and a 3D scene).
- `fairino_model.py` gives the jointless CAD export (`models/fairino_cad.glb`) six
  joints and a `tcp` node, writing `models/fairino_robot.glb`. It also builds the
  clickable floor, `models/fairino_workspace.glb`. If you change it, run
  `uv run python fairino_model.py` to write both files again. A test fails while they
  are out of date.
- `fairino_kinematics.py` is forward and inverse kinematics (damped least squares).
  It reads the joints from the same GLB the viewer draws, so the angles it computes
  match what the viewer shows.

Joint angles are measured from the folded pose the CAD model is drawn in. They are
not the real controller's zero.

## Test

```sh
uv run pytest
```
