import os
import torch
import gradio as gr
import trimesh
import tempfile
import math

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

def rotate(angle, direction, center):
  return trimesh.transformations.rotation_matrix(angle, direction, center)

def generate_mesh(image):
    batch_size = 1
    guidance_scale = 3.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    with open(f'example_mesh.ply', 'wb') as f:
        decode_latent_mesh(xm, latents[0]).tri_mesh().write_ply(f)

    mesh = trimesh.load("example_mesh.ply")

    mesh.apply_transform(rotate(math.pi / 2 * 3, [1, 0, 0], [0, 0, 0]))
    mesh.apply_transform(rotate(math.pi, [0, 1, 0], [0, 0, 0]))

    glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    glb_path = glb_file.name
    mesh.export(glb_path)
    mesh.export('example_mesh.glb')

    del mesh
    del image
    torch.cuda.empty_cache()

    return 'example_mesh.glb'

inputs = gr.inputs.Image(label="Image")
outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
gr.Interface(generate_mesh, inputs, outputs).launch()
