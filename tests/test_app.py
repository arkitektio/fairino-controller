"""The app's actions, called for real: through their ports, with no server.

`call` starts the app as a run starts it and calls an action by name. The arm is
simulated, but it travels at its real speed, so a move takes a few seconds.
"""

import numpy as np
import pytest

import app
import fairino_model


def test_move_to_puts_the_flange_above_the_point(call):
    x, z, height = fairino_model.PEDESTALS["pedestal_1"]
    point = [x, fairino_model.FLOOR_Y + height, z]
    degrees = call("move_to", point=point)
    assert len(degrees) == 6
    tcp = app.CHAIN.forward(np.deg2rad(degrees))[0][:3, 3]
    assert np.allclose(tcp, [point[0], point[1] + app.HOVER, point[2]], atol=1e-3)


def test_move_to_refuses_a_point_out_of_reach(call):
    with pytest.raises(Exception, match="can't reach"):
        call("move_to", point=[2.0, 0.3, 0.0])
