from inspect import cleandoc
import comfy.utils
import torch
import numpy as np


class ReverseLatentBatch:
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "reverselatentbatch"
    CATEGORY = "Tawbaware/latent"
    DESCRIPTION = """
Reverses the order of the latents in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "samples": ("LATENT",),
        },
    } 
    
    def reverselatentbatch(self, samples):
        samples_out = samples.copy()
        samples_out["samples"] = torch.flip(samples["samples"], [0])
        return (samples_out, )
    
    

class LatentBlendGradient:

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"
    CATEGORY = "Tawbaware/latent"
    DESCRIPTION = """
Blends two batches of latents together using a temporal 
gradient.  samples1 is weighted more heavily at start, 
samples2 is weighted more heavily at end.
"""    
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples1": ("LATENT",),
            "samples2": ("LATENT",),
        }}

    def blend(self, samples1, samples2, blend_mode: str="gradient"):

        samples_out = samples1.copy()
        samples1 = samples1["samples"]
        samples2 = samples2["samples"]

        if (samples1.shape[0] != samples2.shape[0]):
            raise ValueError(f"Samples do not have the same length: {samples1.shape[0]} vs {samples2.shape[0]}")

        if samples1.shape != samples2.shape:
            #Tensor shape: [B,C,H,W]
            #Image shape:  [B,H,W,C]
            samples2.permute(0, 3, 1, 2) #Based on code foundin comfyui/nodes
            samples2 = comfy.utils.common_upscale(samples2, samples1.shape[3], samples1.shape[2], 'bicubic', crop='center')
            samples2.permute(0, 2, 3, 1)

        if blend_mode == "normal":
            blend_factor = 0.5
            samples_out["samples"] = samples1 * blend_factor + samples2 * (1-blend_factor)
        elif blend_mode == "gradient":
            #transition from samples1 to samples2
            batch_size = samples1.shape[0]
            for i in range(batch_size):
                blend_factor = (i+0.5)/batch_size
                samples_out["samples"][i] = samples1[i] * (1-blend_factor) + samples2[i] * blend_factor
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")
            
        return (samples_out,)


class WanVideoReCamMasterGenerateOrbitCameraEx:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents": ("LATENT", {"tooltip": "Needed to calculate number of frames"}),
            #"num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to generate"}),
            "degrees_h": ("INT", {"default": 30, "min": -90, "max": 90, "step": 1, "tooltip": "Degrees to orbit horizontally"}),
            "degrees_v": ("INT", {"default": 0, "min": -90, "max": 90, "step": 1, "tooltip": "Degrees to orbit vertically"}),
            "zoom": ("FLOAT", {"default": 0, "min": -10, "max": 0.99, "step": 0.1, "tooltip": "Zoom in (values between 0 and 1) or out (values less than 0)"}),
        },
        }

    RETURN_TYPES = ("CAMERAPOSES",)
    RETURN_NAMES = ("camera_poses",)
    FUNCTION = "process"
    CATEGORY = "Tawbaware/WanVideoWrapper"
    DESCRIPTION ="""
This is a modified and enhanced version of Kijai's 
WanVideoReCamMasterGenerateOrbitCamera custom node 
(https://github.com/kijai/ComfyUI-WanVideoWrapper).  
This node allows for the creation of arbitrary camera 
paths by combining an orbit (rotation in the x/y 
plane around the vertical axis), an elevation change 
(shift up/down along the z axis) and a zoom (towards
or away from subject).
"""

    def process(self, latents, degrees_h, degrees_v, zoom):
        def generate_orbit(num_frames=81, degrees_horizontal=45, degrees_vertical=0, zoom=0):
            camera_data = []
            #Note: camera position and center position were estimated by 
            #analyzing coordinates for arc_* paths in recam_extrinsics.json
            #and comparing to generated videos
            center = np.array([3685, 1380, 240])  #Center point of orbit
            camera = np.array([3390, 1380, 240])  #Initial camera position
            subj_dist = abs(camera[0] - center[0])
            
            for i in range(num_frames):
                # Calculate angles between 0 and specified degrees in both horitontal and vertical directions
                angle_h_rad = np.radians(i * degrees_horizontal / (num_frames - 1))
                angle_v_rad = np.radians(i * degrees_vertical / (num_frames - 1))
                #Zoom in: values between 0 and 1
                #Zoom out: values less than 0
                zoom_factor = (i * zoom / (num_frames - 1))
                
                angle_h_cos = np.cos(angle_h_rad)
                angle_h_sin = np.sin(angle_h_rad)
                angle_v_cos = np.cos(-angle_v_rad)
                angle_v_sin = np.sin(-angle_v_rad)
                angle_v_tan = np.tan(angle_v_rad)

                #First, calculate a translation vector that describes the new location of the camera after motion
                # The x,y coordinates are calculated assuming a circular path around center position (i.e. rotation around z-axis)
                # The z coordinate reflects the vertical height of the camera and is calculated by shifting the position up/down along z-axis
                x = camera[0] + (subj_dist * (1-abs(angle_h_cos)))
                y = camera[1] + (subj_dist * angle_h_sin)
                z = camera[2] + (subj_dist * angle_v_tan)

                #Zoom in/out
                x = x + ((center[0] - x) * zoom_factor)
                y = y + ((center[1] - y) * zoom_factor)
                z = z + ((center[2] - z) * zoom_factor)                
                
                #Second, calculate a rotation matrix that describes the required camera orientation after camera 
                #position change.  This is the combination of two rotations:
                #1. A "horizontal" rotation in x/y plane (i.e. a left/right rotation around z axis)
                #2. A "vertical" rotation in x/z plane (i.e. an up/down rotation around y axis)
                rot1 = np.array([
                    [angle_h_cos, angle_h_sin, 0],
                    [-angle_h_sin, angle_h_cos, 0],
                    [0, 0, 1]
                ])
                
                rot2 = np.array([
                    [angle_v_cos, 0, -angle_v_sin],
                    [0, 1, 0],
                    [angle_v_sin, 0, angle_v_cos]
                ])
                
                #combine the two rotations into a single rotation matrix
                rot = np.matmul(rot2, rot1)
                
                #combine rotation matrix and translation vector into single camera matrix
                transform = np.array([
                    [1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]
                ])
                transform[0:3, 0:3] = rot
                
                camera_data.append(transform)
               
            return camera_data
                
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape
        num_frames = (T - 1) * 4 + 1

        # Generate orbit data
        camera_transforms = generate_orbit(num_frames=num_frames, degrees_horizontal=degrees_h, degrees_vertical=degrees_v, zoom=zoom)
        
        traj = camera_transforms[::4]
        traj = np.stack(traj)
       
        return (traj,)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    #"Example": Example,
    "ReverseLatentBatch": ReverseLatentBatch,
    "LatentBlendGradient": LatentBlendGradient,
    "WanVideoReCamMasterGenerateOrbitCameraEx": WanVideoReCamMasterGenerateOrbitCameraEx,
}

# A dictionary that contains the friendly/human readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    #"Example": "Example Node",
    "ReverseLatentBatch": "Reverse Latent Batch",
    "LatentBlendGradient": "Latent Blend Gradient",
    "WanVideoReCamMasterGenerateOrbitCameraEx": "WanVideo ReCamMaster Generate Orbit Camera Ex"
}
