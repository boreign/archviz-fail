#!/Users/bbm/miniconda3/envs/guabia_env/bin/python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    ControlNetModel, 
    UniPCMultistepScheduler
)
# Import from_single_file is no longer needed if all models are HF folders
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import fire
import os

def render(input_path: str, output_path: str,
           style_prompt: str,
           seed: int = 44,
           denoise: float = 0.6, # Increased denoise for more transformation
           steps: int = 40,     # Increased steps
           ref_steps: int = 10, # Increased refiner steps
           ref_denoise: float = 0.3, # Slightly increased refiner denoise
           cfg: float = 7.0,    # Increased cfg for stronger prompt adherence
           canny_scale: float = 1.0, # Controlnet scale for canny
           depth_scale: float = 1.0, # Controlnet scale for depth
           w: int = 512,
           h: int = 512):
    """
    renders a sketchup image to photoreal using sdxl + controlnets

    args:
        input_path (str): path to the input sketchup image.
        output_path (str): path to save the output image.
        style_prompt (str): text describing the desired style (e.g., "modern living room").
        seed (int): random seed.
        denoise (float): img2img denoise strength for base model (0.0 to 1.0).
        steps (int): number of inference steps for base model.
        ref_steps (int): number of inference steps for refiner.
        ref_denoise (float): img2img denoise strength for refiner (0.0 to 1.0).
        cfg (float): classifier-free guidance scale.
        canny_scale (float): strength of the canny controlnet.
        depth_scale (float): strength of the depth controlnet.
        w (int): output image width.
        h (int): output image height.
    """

    device = "mps"
    if not torch.backends.mps.is_available():
        print("!!!! mps not available. check torch installation or hardware.")
        raise environmenterror("mps not available, cannot run on this device.")

    # Define base model directory
    MODEL_BASE_DIR = "/Users/bbm/Documents/ai/models/img" # NEW BASE PATH

    # Define model paths relative to the base directory (assuming HF folder structure)
    # Base model (now HF folder)
    base_model_folder = os.path.join(MODEL_BASE_DIR, "RealVisXL_V5.0") # UPDATED PATH
    # Refiner model (HF folder)
    refiner_model_folder = os.path.join(MODEL_BASE_DIR, "stable-diffusion-xl-refiner-1.0") # UPDATED PATH
    # ControlNet models (HF folder structure)
    canny_model_folder = os.path.join(MODEL_BASE_DIR, "controlnet-canny-sdxl-1.0") # UPDATED PATH
    depth_model_folder = os.path.join(MODEL_BASE_DIR, "controlnet-depth-sdxl-1.0") # UPDATED PATH


    # check if paths exist and are directories
    for p in [base_model_folder, refiner_model_folder, canny_model_folder, depth_model_folder]:
        resolved_path = os.path.expanduser(p)
        if not os.path.exists(resolved_path):
             print(f"!!!! model path not found: {resolved_path}")
             raise filenotfounderror(f"model path not found: {resolved_path}")
        if not os.path.isdir(resolved_path):
             print(f"!!!! model path expected to be a folder but is not: {resolved_path}")
             raise isadirectoryerror(f"model path expected to be a folder: {resolved_path}")


    print(f"loading base model from folder: {base_model_folder}")
    # *** CHANGED back to from_pretrained for base model ***
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        os.path.expanduser(base_model_folder),
       # torch_dtype=torch.float16,
        low_cpu_mem_usage=True,      # ← streams chunks in/out of CPU
        use_safetensors=True # Assuming HF repo uses safetensors
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"loading controlnets from folders: {canny_model_folder}, {depth_model_folder}")
    # ControlNets still loaded from folders using from_pretrained
    canny = ControlNetModel.from_pretrained(
        os.path.expanduser(canny_model_folder),
       # torch_dtype=torch.float16
    ).to(device)
    depth = ControlNetModel.from_pretrained(
        os.path.expanduser(depth_model_folder),
        #torch_dtype=torch.float16
    ).to(device)
    # pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    pipe.controlnet = [canny, depth]
    cn_scales = [canny_scale, depth_scale]

    # load & preprocess image
    print(f"loading input image: {input_path}")
    init_image = load_image(input_path).convert("RGB")


    # define prompts
    pos_prompt_base = f"a photorealistic 3D render of this environment,{style_prompt}, minimal wood grain, smooth wooden surfaces, cinematic lighting, ultra-detailed 3D render, photorealistic, octane lighting, V-Ray style"
    neg_prompt = "cartoon, dreamy, watermark, anime, distorted, painting, highly textured wood, wood grain, knots, rough surfaces, detailed wood texture, (maximalist:1.3), ornate, busy pattern"

    print("generating base image...")
    # base pass img2img with controlnets
    img = pipe(
        prompt=pos_prompt_base,
        negative_prompt=neg_prompt,
        image=init_image, # feed original image, diffusers handles preproc with controlnets
        strength=denoise,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=w, height=h,
        controlnet_conditioning_scale=cn_scales
    ).images[0]

    img.save('/Users/bbm/Documents/8_apps/guabia_engine/assets/output/debug_base.png')

    # refiner pipeline - FROM HUGGING FACE FOLDER
    print(f"loading refiner model from folder: {refiner_model_folder}")
    ref_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        os.path.expanduser(refiner_model_folder),
        #torch_dtype=torch.float16,
        low_cpu_mem_usage=True,      # ← streams chunks in/out of CPU
        use_safetensors=True # Assuming HF repo uses safetensors
    ).to(device)
    ref_pipe.scheduler = UniPCMultistepScheduler.from_config(ref_pipe.scheduler.config)
    #ref_pipe.enable_model_cpu_offload()
    ref_pipe.enable_attention_slicing()


    print("generating refined image...")
    # refiner pass
    out = ref_pipe(
        prompt=pos_prompt_base,
        negative_prompt=neg_prompt,
        image=img,
        strength=ref_denoise,
        guidance_scale=cfg,
        num_inference_steps=ref_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=w, height=h
    ).images[0]

    print(f"saving output to: {output_path}")
    out.save(output_path)
    print("done.")


if __name__ == "__main__":
    fire.Fire(render)
