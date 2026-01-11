import pytorch
from diffusers import StableDiffusionPipeline 

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token="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") 
pipe.to(device) 

prompt = """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin,
extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans
"""

image = pipe(prompt).images[0]

st.image(image, caption="My Photo", use_container_width=True)
