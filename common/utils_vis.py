import pickle
import numpy as np
import torch
import math
import sys
import os
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.manolayer import ManoLayer

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader
)


class Renderer():
    def __init__(self, img_size, device='cpu'):
        self.img_size = img_size
        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        self.amblights = AmbientLights(device=device)
        self.point_lights = PointLights(location=[[0, 0, -1.0]], device=device)

        self.renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=HardPhongShader(device=device)
        )
        self.device = device

    def build_camera(self, cameras=None,
                     scale=None, trans2d=None):
        if scale is not None and trans2d is not None:
            bs = scale.shape[0]
            R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
            T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
            return OrthographicCameras(focal_length=2 * scale.to(self.device),
                                       principal_point=-trans2d.to(self.device),
                                       R=R.to(self.device),
                                       T=T.to(self.device),
                                       in_ndc=True,
                                       device=self.device)
        if cameras is not None:
            # cameras: bs x 3 x 3
            fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / self.img_size
            pps = -cameras[:, :2, -1] * 2 / self.img_size + 1
            return PerspectiveCameras(focal_length=fs.to(self.device),
                                      principal_point=pps.to(self.device),
                                      in_ndc=True,
                                      device=self.device
                                      )

    def build_texture(self, uv_verts=None, uv_faces=None, texture=None,
                      v_color=None):
        if uv_verts is not None and uv_faces is not None and texture is not None:
            return TexturesUV(texture.to(self.device), uv_faces.to(self.device), uv_verts.to(self.device))
        if v_color is not None:
            return TexturesVertex(verts_features=v_color.to(self.device))

    def render(self, verts, faces, cameras, textures, amblights=False,
               lights=None):
        if lights is None:
            if amblights:
                lights = self.amblights
            else:
                lights = self.point_lights
        mesh = Meshes(verts=verts.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        alpha = output[..., 3]
        img = output[..., :3] / 255
        return img, alpha

