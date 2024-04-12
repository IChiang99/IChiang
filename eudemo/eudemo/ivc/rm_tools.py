# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
EU-DEMO Equatorial Port
"""
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Optional, Tuple, Union

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.base.reactor_config import ConfigParams
from bluemira.geometry.tools import boolean_cut, boolean_fuse, force_wire_to_spline, make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace


def scale_geometry(profile:BluemiraWire, scale:float, dir:str = 'x', origin:list = [0., 0., 0.], _closed = True) -> BluemiraWire:
    """
    Scales a given BluemiraWire by a given scale factor along an axis from a point

    Outputs a scaled BluemiraWire
    """
    points = profile.discretize(600, byedges=True)
    x = points.x
    y = points.y
    z = points.z
    if dir == 'x':
        datum = origin[0]
        x = [(scale * (i - datum)) + datum for i in x]
    if dir == 'y':
        datum = origin[1]
        y = [(scale * (i - datum)) + datum for i in y]
    if dir == 'z':
        datum = origin[2]
        z = [(scale * (i - datum)) + datum for i in z]
    new_profile = BluemiraWire(make_polygon({"x": x, "y": y, "z": z}, closed=_closed))
    return new_profile

def face_the_wire(profile:BluemiraWire) -> BluemiraFace:
    """ Converts profile wire to face """
    return BluemiraFace([force_wire_to_spline(profile)])


def vv_koz_modifier(inner_wire: BluemiraWire, theta: float) -> BluemiraWire:
    """ 
    Un-curves the lower-outboard part of the vv_koz (xz-plane) into a sharp angle

    Parameters
    ----------
    inner_wire:
        BM Wire of vessel koz (void space)
    theta:
        Angle of segment of inner wire to modify, from vertical down

    Returns
    -------
    BM wire of modified profile in the x-z plane

    """

    # Determine ivc_koz COM
    R_0 = inner_wire.center_of_mass
    # Draw an x-z wedge of the vessel to cut out of ivc_koz wire, centred on x-section COM
    bounding_box = inner_wire.bounding_box
    z = bounding_box.z_min
    height = abs(R_0[2] - z)
    x = (height * np.tan(theta * (np.pi/180))) + R_0[0]
    cut_out_points = np.array(
        [
            [R_0[0], 0, R_0[2]],
            [R_0[0], 0, z],
            [x, 0, z],
            [R_0[0], 0, R_0[2]],
        ]
    )
    cut_zone = make_polygon(cut_out_points, closed=True)
    # Cut the triangle out of the ivc_koz box
    ivc_koz_wires = boolean_cut(inner_wire, [cut_zone])

    # Select the correct wire cut-out (highest COG z-co'ord)
    new_wire = BluemiraWire(
        ivc_koz_wires[np.argmax(
            [w.center_of_mass[2] for w in ivc_koz_wires]
            )])
    # Find the start and end wire-edges
    edge_a = new_wire.edges[0]
    edge_b = new_wire.edges[-1]
    a_start = edge_a.start_point()
    b_end = edge_b.end_point()
    # Find the start/end wire-edge gradients and z-intercepts
    a_end = edge_a.end_point()
    b_start = edge_b.start_point()
    m_a = (a_end[2][0] - a_start[2][0])/(a_end[0][0] - a_start[0][0])
    m_b = (b_end[2][0] - b_start[2][0])/(b_end[0][0] - b_start[0][0])
    c_a = a_start[2][0] - (m_a * a_start[0][0])
    c_b = b_start[2][0] - (m_b * b_start[0][0])
    # Find the [x, 0, z] of the a-b intercept
    i_x = (c_b - c_a)/(m_a - m_b)
    i_z = (m_a * i_x) + c_a
    # TODO: should check also i_z == (m_b * i_x) + c_b

    # Draw the new_extension (two wires) that make a-i-b - interpolated midpoint is magic point to make it work?
    new_extension = BluemiraWire(
        make_polygon([
            [a_start[0][0] , 0., a_start[2][0]],
            [(a_start[0][0]+i_x)/2, 0., (a_start[2][0]+i_z)/2],
            [i_x, 0., i_z], 
            [b_end[0][0], 0., b_end[2][0]],
            ],
            ))

    # x_points = [i.start_point().x[0] for i in new_wire.edges]
    # x_points.append(new_wire.edges[-1].end_point().x[0])
    # z_points = [i.start_point().z[0] for i in new_wire.edges]
    # z_points.append(new_wire.edges[-1].end_point().z[0])
    # plt.plot(x_points, z_points, c = 'orange', ls = '-')
    # x_points = [i.start_point().x[0] for i in new_extension.edges]
    # x_points.append(new_extension.edges[-1].end_point().x[0])
    # z_points = [i.start_point().z[0] for i in new_extension.edges]
    # z_points.append(new_extension.edges[-1].end_point().z[0])
    # plt.plot(x_points, z_points, c = 'grey', ls = '-.')

    # plt.savefig("VESSEL.png")

    # Patch the new_extension onto the cut-out koz wire
    output = BluemiraWire([new_wire, new_extension])
    return output

class BlanketChimney_Designer(Designer[Tuple[BluemiraWire, float]]):
    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        height: float,

    ):
        super().__init__(params, build_config)

    
    def run(self):
        # return chimney_profile_xz, angle
        pass

