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
EUDEMO builder for blanket
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Union

import numpy as np

from bluemira.base.builder import Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    circular_pattern_component,
    get_n_sectors,
    linear_pattern,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.solid import BluemiraGeo, BluemiraSolid
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    offset_wire,
    slice_shape,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.ivc.rm_tools import face_the_wire, scale_geometry


@dataclass
class BlanketBuilderParams(ParameterFrame):
    """
    Blanket builder parameters
    """

    n_TF: Parameter[int]
    n_bb_inboard: Parameter[int]
    n_bb_outboard: Parameter[int]
    c_rm: Parameter[float]


class BlanketBuilder(Builder):
    """
    Blanket builder

    Parameters
    ----------
    params:
        the parameter frame
    build_config:
        the build config
    ib_silhouette:
        inboard blanket silhouette
    ob_silhouette:
        outboard blanket silhoutte
    bb_silhouette:
        breeding blanket silhouette
    split_geom:
        IB/OB splitter geom
    vv_void:
        VV voidspace silhouette
    chimney_profiles
        top-down chimney profiles
    """

    BB = "BB"
    IBS = "IBS"
    OBS = "OBS"
    param_cls: Type[BlanketBuilderParams] = BlanketBuilderParams
    params: BlanketBuilderParams

    def __init__(
        self,
        params: Union[BlanketBuilderParams, Dict],
        build_config: Dict,
        ib_silhouette: BluemiraFace,
        ob_silhouette: BluemiraFace,
        bb_silhouette: BluemiraFace,
        split_geom: BluemiraFace,
        vv_void: BluemiraWire,
        chimney_profiles: Union[BluemiraFace, List[BluemiraFace]] = [],
    ):
        super().__init__(params, build_config)
        self.ib_silhouette = ib_silhouette
        self.ob_silhouette = ob_silhouette
        self.bb_silhouette = bb_silhouette
        self.split_face = split_geom
        self.vv_void = vv_void
        self.profiles = chimney_profiles

    def build(self) -> Component:
        """
        Build the blanket component.
        """
        segments = self.get_segments(self.ib_silhouette, self.ob_silhouette)
        self.vv_void.close()
        solid_void = pattern_revolved_silhouette(
            BluemiraFace([offset_wire(self.vv_void, self.params.c_rm.value)]), 
            1, 
            16, 
            self.params.c_rm.value
        )
        vv_void = solid_void[0]
        vv_void.rotate(degree=-180 / self.params.n_TF.value)
        chimney_solids = self.chimney_builder(
            self.profiles,
            vv_void,
        )
        whole = self.get_whole_3D(self.bb_silhouette, chimney_solids)
        xyz = self.build_new_xyz(whole, degree=0)
        return self.component_tree(
            xz = [self.build_xz(self.ib_silhouette, self.ob_silhouette)],
            xy = self.build_xy(xyz[0].children[0].children),
            # xy = self.build_xy(segments)
            # xyz = self.build_xyz(segments, degree=0),
            xyz = xyz,
        )

    def build_xz(self, ibs_silhouette: BluemiraFace, obs_silhouette: BluemiraFace):
        """
        Build the x-z components of the blanket.
        """
        ibs = PhysicalComponent(self.IBS, ibs_silhouette)
        obs = PhysicalComponent(self.OBS, obs_silhouette)
        apply_component_display_options(ibs, color=BLUE_PALETTE[self.BB][0])
        apply_component_display_options(obs, color=BLUE_PALETTE[self.BB][1])
        return Component(self.BB, children=[ibs, obs])

    def build_xy(self, segments: List[PhysicalComponent]):
        """
        Build the x-y components of the blanket.
        """
        xy_plane = BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])

        slices = []
        for i, segment in enumerate(segments):
            single_slice = PhysicalComponent(
                segment.name, BluemiraFace(slice_shape(segment.shape, xy_plane)[0])
            )
            apply_component_display_options(single_slice, color=BLUE_PALETTE[self.BB][i])
            slices.append(single_slice)

        return circular_pattern_component(
            Component(self.BB, children=slices), self.params.n_TF.value
        )

    def build_xyz(self, segments: List[PhysicalComponent], degree: float = 360.0):
        """
        Build the x-y-z components of the blanket.
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        # TODO: Add blanket cuts properly in 3-D
        return circular_pattern_component(
            Component(self.BB, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

    def build_new_xyz(self, whole: PhysicalComponent, degree: float = 360.0):
        """
        Create and planar segment the whole sector blanket in 3D
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)
        # make the IB/OB split volume
        split_vec = self.split_face.normal_at()
        foo = extrude_shape(self.split_face, split_vec * VERY_BIG)
        bar = extrude_shape(self.split_face, split_vec * -VERY_BIG)
        cut_tool = boolean_fuse([foo, bar])
        # apply the split volume (split ib from ob)
        new_ib, new_ob = boolean_cut(whole.shape, cut_tool)
        # make the segments
        ib_width = self.calculate_width(new_ib.bounding_box.x_max, sector_degree)
        ob_width = self.calculate_width(
            self.bb_silhouette.bounding_box.x_max, sector_degree
        )
        ibs = self.segment_blanket(new_ib, self.params.n_bb_inboard.value, ib_width)
        obs = self.segment_blanket(new_ob, self.params.n_bb_outboard.value, ob_width)
        # print("There are ",len(ibs),"IBS and n_OBS:", len(obs))
        segments = []
        for name, base_no, bs_shape in [
            [
                self.IBS,
                0,
                ibs,
            ],  # update new_ib & new_ob with split segments list var name
            [self.OBS, self.params.n_bb_inboard.value + 1, obs],
        ]:
            for no, shape in enumerate(bs_shape):
                segment = PhysicalComponent(f"{name}_{no}", shape)
                apply_component_display_options(
                    segment, color=BLUE_PALETTE[self.BB][base_no + no]
                )
                segments.append(segment)

        return circular_pattern_component(
            Component(self.BB, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

    def get_segments(self, ibs_silhouette: BluemiraFace, obs_silhouette: BluemiraFace):
        """
        Create segments of the blanket from inboard and outboard silhouettes
        """
        ibs_shapes = pattern_revolved_silhouette(
            ibs_silhouette,
            self.params.n_bb_inboard.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        obs_shapes = pattern_revolved_silhouette(
            obs_silhouette,
            self.params.n_bb_outboard.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
            # seg_split_geom = 'parallel',      # TODO: put into params properly
            # proportion = 0.5,
        )

        segments = []
        for name, base_no, bs_shape in [
            [self.IBS, 0, ibs_shapes],
            [self.OBS, self.params.n_bb_inboard.value + 1, obs_shapes],
        ]:
            for no, shape in enumerate(bs_shape):
                segment = PhysicalComponent(f"{name}_{no}", shape)
                apply_component_display_options(
                    segment, color=BLUE_PALETTE[self.BB][base_no + no]
                )
                segments.append(segment)
        return segments

    def chimney_builder(
        self, xy_profiles: Union[BluemiraFace, List[BluemiraFace]], vv_void: BluemiraSolid
    ) -> List[BluemiraSolid]:
        """Turns chimney xy profiles into a list of chimney solids"""
        if isinstance(xy_profiles, BluemiraFace):
            xy_profiles = [xy_profiles]
        chimneys = []
        for face in xy_profiles:
            # make faces into prismatic solids
            height = vv_void.center_of_mass[2] - face.center_of_mass[2]
            vec = (0.0, 0.0, height)
            column = extrude_shape(face, vec)
            chimney = boolean_cut(column, [vv_void])
            chimneys.extend(chimney)
        return chimneys

    def calculate_width(self, radius: float, angle: float) -> float:
        return np.sqrt(2) * radius * np.sqrt(1 - np.cos(np.deg2rad(angle)))

    def segment_blanket(
        self, shape: BluemiraGeo, n_segments: int, bb_width: float, angle: float = 0
    ) -> List[BluemiraGeo]:
        """
        Segments a blanket solid using parallel radial cuts
        """
        c_rm = self.params.c_rm.value
        half_c = -c_rm / 2
        # set up the cut-tool
        x = [0, VERY_BIG, VERY_BIG, 0]
        y = [half_c, half_c, half_c, half_c]
        z = [-VERY_BIG, -VERY_BIG, VERY_BIG, VERY_BIG]
        foo = BluemiraFace(
            make_polygon({"x": x, "y": y, "z": z}, closed=True)
        )  # Make a big xz-cut tool
        xz_cut_tool = extrude_shape(foo, (0.0, c_rm, 0.0))
        # # xz_cut_tool.rotate(degree = half_a)     # Rotate a half-sector around
        # set up the translation vector
        # unit_vec = [
        #     -np.sin(np.deg2rad(half_a)),
        #     np.cos(np.deg2rad(half_a)),
        #     0.]
        unit_vec = [0.0, 1.0, 0.0]
        spacing_vec = [i * (bb_width / n_segments) for i in unit_vec]
        translate_vec = [i * (-bb_width / 2) for i in unit_vec]
        # generate cut tools and perform the segmentation
        xz_cut_tool.translate(tuple(translate_vec))
        cut_tools = linear_pattern(xz_cut_tool, tuple(spacing_vec), n_segments)
        segments = boolean_cut(shape, cut_tools)
        return segments

    def get_whole_3D(
        self, whole_profile: BluemiraFace, chimneys: List[BluemiraGeo] = []
    ):
        """
        Get an unsplit blanket component with chimneys
        """
        solid = pattern_revolved_silhouette(
            whole_profile,
            1,
            16,
            self.params.c_rm.value,
        )

        shape = solid[0]
        shape.rotate(degree=-180 / self.params.n_TF.value)
        for chimney in chimneys:
            shape = boolean_fuse([shape, chimney])
        sector = PhysicalComponent("BB", shape)
        apply_component_display_options(sector, color=BLUE_PALETTE[self.BB][0])
        return sector
