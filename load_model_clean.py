from tqdm import tqdm
from gaussianimage_rs import GaussianImage_RS
from gsplat import project_gaussians_2d_scale_rot
from gsplat.project_gaussians_2d import project_gaussians_2d
import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from train import image_path_to_tensor
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from gaussianimage_cholesky import GaussianImage_Cholesky
from einops import rearrange

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

image_path = '/home/vitor/Documents/doc/GaussianImage/dataset/neutral_front/001_03.jpg'
# model_path = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_Cholesky_50000_30000/001_03/gaussian_model.pth.tar'
model_path = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/gaussian_model.pth.tar'
# model_path = 'checkpoints/teste/teste/gaussian_model_0.99.pth.tar'
device = torch.device("cuda:0")
num_points = 30_000
BLOCK_H, BLOCK_W = 16, 16
gt_image = image_path_to_tensor(image_path).to(device)
H, W = gt_image.shape[2], gt_image.shape[3]


# gaussian_model = GaussianImage_Cholesky(
#     loss_type="L2", 
#     opt_type="adan", 
#     num_points=num_points, 
#     H=H, 
#     W=W, 
#     BLOCK_H=BLOCK_H, 
#     BLOCK_W=BLOCK_W, 
#     device=device, 
#     lr=1e-3, 
#     quantize=False).to(device)

gaussian_model = GaussianImage_RS(
    loss_type="L2", 
    opt_type="adan", 
    num_points=num_points, 
    H=H, 
    W=W, 
    BLOCK_H=BLOCK_H, 
    BLOCK_W=BLOCK_W, 
    device=device, 
    lr=1e-3, 
    quantize=False).to(device)

print(f"loading model path:{model_path}")
checkpoint = torch.load(model_path, map_location=device)
model_dict = gaussian_model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(pretrained_dict)
gaussian_model.load_state_dict(model_dict)


gaussian_model.eval()
with torch.no_grad():
    out = gaussian_model()
    out_rgb = rearrange(out['render'], '1 c h w -> h w c').cpu().numpy()
    


    

def plot_conics_plotly():
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
        gaussian_model.get_xyz, 
        gaussian_model.get_cholesky_elements, 
        gaussian_model.H, 
        gaussian_model.W, 
        gaussian_model.tile_bounds)

    H = gaussian_model.H  
    W = gaussian_model.W  


    fig = go.Figure()
    min_ = gaussian_model.get_features.min()
    max_ = gaussian_model.get_features.max()
    rgb = torch.floor(((gaussian_model.get_features-min_)/(max_-min_)) * 255).detach().cpu().numpy().astype(np.uint8)

    amount_to_plot = 30_000

    for i, (xy, radius, conic, rgb) in tqdm(enumerate(zip(xys, radii, conics,rgb)), total=amount_to_plot):
        if i>amount_to_plot:break

        # Get center (x, y) and conic parameters [A, B, C]
        mu_x, mu_y = xy.detach().cpu().numpy()  # Gaussian center
        A, B, C = conic.detach().cpu().numpy()  # Conic parameters

        # Compute the semi-major and semi-minor axes using the conic equation
        # The eigenvalues of the matrix defined by [A, B, C] give the axes lengths of the ellipse.
        # We first construct the conic matrix.
        conic_matrix = np.array([[A, B], [B, C]])
        
        # Eigen decomposition to get the axes of the ellipse
        eigvals, eigvecs = np.linalg.eig(conic_matrix)

        # Eigenvalues are the inverses of the squared axes lengths (semi-major and semi-minor)
        semi_major = np.sqrt(1 / eigvals[0])  # Larger axis
        semi_minor = np.sqrt(1 / eigvals[1])  # Smaller axis

        # Ellipse orientation (rotation angle)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        # Create points for the ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = semi_major * np.cos(theta)
        y_ellipse = semi_minor * np.sin(theta)

        # Rotate the ellipse based on the orientation angle
        x_rot = x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle)
        y_rot = x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle)

        # Translate the ellipse to its center (mu_x, mu_y)
        x_final = mu_x + x_rot
        y_final = mu_y + y_rot

        
        r,g,b = rgb
        fig.add_trace(go.Scatter(
            x=x_final,
            y=y_final,
            mode='lines',
            fill='toself', 
            fillcolor=f'rgba({r}, {g}, {b}, 1)',  
            line=dict(color=f'rgba({r}, {g}, {b}, 1)', width=2),
            name=f'Gaussian {i+1}',  
        ))

    fig.update_layout(
        title="2D Gaussian Projections on HxW Grid",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        width=1200,
        height=900,
        xaxis=dict(range=[0, W]),  
        yaxis=dict(
            range=[0, H],  
            autorange='reversed'  
        ), 
    )

    # Show the interactive plot
    fig.show()

from errno import EEXIST
from os import makedirs, path
import os
from plyfile import PlyData, PlyElement

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def save_ply(path):
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            gaussian_model.get_xyz, 
            gaussian_model.get_scaling, 
            gaussian_model.get_rotation,
            gaussian_model.H, 
            gaussian_model.W, 
            gaussian_model.tile_bounds)
    

    xyz = xys.detach().cpu().numpy()
    xyz = np.concat([xyz,np.zeros((xys.shape[0],1))],axis=-1)
    normals = np.zeros((xys.shape[0],45)) #+ 1e-3
    f_dc = gaussian_model.get_features.detach().contiguous().cpu().numpy() #+ 1e-8 #.transpose(1, 2).flatten(start_dim=1)
    # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = np.zeros_like(f_dc) #+ 1
    opacities = gaussian_model.get_opacity.detach().cpu().numpy()
    scale = gaussian_model.get_scaling.detach().cpu().numpy() #/ 2
    rotation = gaussian_model.get_rotation.detach().cpu().numpy()


    # testing...
    scale_add = np.zeros((xys.shape[0],1)) #+ 1e-8
    scale = np.concat([scale,scale_add],axis=-1)

    R = np.array([
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ], dtype=np.float32) for theta in rotation.flatten()
    ], dtype=np.float32)
    q = np.zeros((R.shape[0], 4))

    # Thanks daniel perazzo
    # Compute the trace of the matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Small epsilon value to avoid division by zero
    epsilon = 1e-8

    # Quaternion conversion
    for i, mat in enumerate(R):
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        
        if trace > epsilon:  # Check if the trace is sufficiently large
            q_w = 0.5 * np.sqrt(1 + trace)
            q_x = (mat[2, 1] - mat[1, 2]) / (4 * q_w)
            q_y = (mat[0, 2] - mat[2, 0]) / (4 * q_w)
            q_z = (mat[1, 0] - mat[0, 1]) / (4 * q_w)
            q[i] = [q_w, q_x, q_y, q_z]
        else:
            # Special case when the trace is near zero (possible 180-degree rotation)
            # You can handle this by using a different formula, like from the matrix diagonal
            q_w = np.sqrt(max(0.5 * (1 + mat[0, 0] + mat[1, 1] + mat[2, 2]),0))
            q_x = np.sqrt(max(0.5 * (1 + mat[0, 0] - mat[1, 1] - mat[2, 2]),0)) if q_w != 0 else 0
            q_y = np.sqrt(max(0.5 * (1 - mat[0, 0] + mat[1, 1] - mat[2, 2]),0)) if q_w != 0 else 0
            q_z = np.sqrt(max(0.5 * (1 - mat[0, 0] - mat[1, 1] + mat[2, 2]),0)) if q_w != 0 else 0
            q[i] = [q_w, q_x, q_y, q_z]


    rotation = q + 1e-8
   

    dtype_full = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
        ('f_dc_r', 'f4'),
        ('f_dc_g', 'f4'),
        ('f_dc_b', 'f4'),
        ('f_rest_0','f4'),
        ('f_rest_1','f4'),
        ('f_rest_2','f4'),
        ('f_rest_3','f4'),
        ('f_rest_4','f4'),
        ('f_rest_5','f4'),
        ('f_rest_6','f4'),
        ('f_rest_7','f4'),
        ('f_rest_8','f4'),
        ('f_rest_9','f4'),
        ('f_rest_10','f4'),
        ('f_rest_11','f4'),
        ('f_rest_12','f4'),
        ('f_rest_13','f4'),
        ('f_rest_14','f4'),
        ('f_rest_15','f4'),
        ('f_rest_16','f4'),
        ('f_rest_17','f4'),
        ('f_rest_18','f4'),
        ('f_rest_19','f4'),
        ('f_rest_20','f4'),
        ('f_rest_21','f4'),
        ('f_rest_22','f4'),
        ('f_rest_23','f4'),
        ('f_rest_24','f4'),
        ('f_rest_25','f4'),
        ('f_rest_26','f4'),
        ('f_rest_27','f4'),
        ('f_rest_28','f4'),
        ('f_rest_29','f4'),
        ('f_rest_30','f4'),
        ('f_rest_31','f4'),
        ('f_rest_32','f4'),
        ('f_rest_33','f4'),
        ('f_rest_34','f4'),
        ('f_rest_35','f4'),
        ('f_rest_36','f4'),
        ('f_rest_37','f4'),
        ('f_rest_38','f4'),
        ('f_rest_39','f4'),
        ('f_rest_40','f4'),
        ('f_rest_41','f4'),
        ('f_rest_42','f4'),
        ('f_rest_43','f4'),
        ('f_rest_44','f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'),
        ('scale_1', 'f4'),
        ('scale_2', 'f4'),
        ('rot_1', 'f4'),
        ('rot_2', 'f4'),
        ('rot_3', 'f4'),
        ('rot_4', 'f4'),
    ] 
    elements = np.empty((xyz.shape[0],len(dtype_full)), dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    attributes = attributes.reshape(-1, len(dtype_full))
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


save_ply('/home/vitor/Documents/IMPA/gaussian-splatting/output/test/point_cloud/iteration_7000/point_cloud.ply')

# save_ply('/home/vitor/Documents/IMPA/gaussian-splatting/output/test_joao_099/point_cloud/iteration_7000/point_cloud.ply')
