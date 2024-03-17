from flask import Flask, render_template
from gradio_client import Client

app = Flask(__name__)
client = Client("prodia/fast-stable-diffusion")

@app.route('/')
def generate_and_display_image():
    prompt = "Design a minimalist sitting room that exudes tranquility and elegance. Incorporate room decor items that are visually captivating and evoke a sense of emotional resonance. Your design should prioritize simplicity, clean lines, and a harmonious color palette. Consider how each element contributes to creating a serene and inviting atmosphere for relaxation and contemplation"
    negative_prompt = "3d, cartoon, anime, eformed eyes, deformed nose, deformed ears, deformed nose, bad anatomy, ugly, extra limb, low resolution, pixelated, distorted proportions, unnatural colors, unrealistic lighting, excessive noise, unrealistic shadows, unnatural poses"
    stable_diffusion_checkpoint = "amIReal_V41.safetensors [0a8a2e61]"
    sampling_steps = 20.0 
    sampling_method = "DPM++ 2M Karras"
    cfg_scale = 7.0
    width = 512
    height = 512
    seed = -1.0

    result = client.predict(
        prompt,
        negative_prompt,
        stable_diffusion_checkpoint,
        sampling_steps,
        sampling_method,
        cfg_scale,
        width,
        height,
        seed,
        api_name="/txt2img"
    )

    return render_template('index.html', image=result)

if __name__ == '__main__':
    app.run(debug=True)
