import os
import torch
import discord
from discord.ext import commands
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

class MyBot(commands.Bot):
    async def on_ready(self):
        await self.tree.sync()
        print(f'Logged in as {self.user}')

bot = MyBot(command_prefix='/', intents=intents)
executor = ThreadPoolExecutor()

# Function to load a model and move it to float16 and CUDA in segments
def load_model_to_cuda(model_class, pretrained_model_name_or_path, subfolder=None):
    if subfolder:
        model = model_class.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
    else:
        model = model_class.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.float16)
    torch.cuda.empty_cache()  # Clear the cache before moving to CUDA
    model = model.to("cuda", non_blocking=True)
    return model

# Load Stable Diffusion pipeline dynamically without the safety checker
def load_pipeline(model_name):
    model_path = os.path.join("models", model_name)
    
    # Check if the model directory exists and load the model from there
    if os.path.isdir(model_path):
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        # Load the default model if custom model doesn't exist
        text_encoder = load_model_to_cuda(CLIPTextModel, "openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        unet = load_model_to_cuda(UNet2DConditionModel, "CompVis/stable-diffusion-v1-4", subfolder="unet")
        vae = load_model_to_cuda(AutoencoderKL, "CompVis/stable-diffusion-v1-4", subfolder="vae")
        scheduler = DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=None,
        ).to("cuda")

    return pipeline

# Generate image function to be run in a separate thread
def generate_image(prompt, model, height, width, steps, seed):
    pipeline = load_pipeline(model)
    generator = torch.Generator("cuda").manual_seed(seed)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    # Ensure all inputs are in float16
    pipeline.unet.to(torch.float16)
    pipeline.vae.to(torch.float16)
    pipeline.text_encoder.to(torch.float16)
    
    # Reduce memory footprint by limiting the height and width
    height = min(height, 512)
    width = min(width, 512)
    
    image = pipeline(prompt, height=height, width=width, num_inference_steps=steps, generator=generator).images[0]

    # Save the image
    image_path = "output.png"
    image.save(image_path)

    # Clear GPU memory after generating the image
    del pipeline
    torch.cuda.empty_cache()

    return image_path

@bot.tree.command(name='diffuse', description='Generate an image using Stable Diffusion')
async def diffuse(interaction: discord.Interaction, prompt: str, model: str = "default", height: int = 512, width: int = 512, steps: int = 50):
    await interaction.response.send_message(f"Generating image for prompt: {prompt} using model: {model} with dimensions {width}x{height} and {steps} steps.")

    # Generate a random seed
    seed = random.randint(0, 999999)

    # Run the image generation in a separate thread
    loop = asyncio.get_running_loop()
    image_path = await loop.run_in_executor(executor, generate_image, prompt, model, height, width, steps, seed)

    # Send the image to the user
    await interaction.followup.send(file=discord.File(image_path))

@bot.tree.command(name='list_models', description='List all available models')
async def list_models(interaction: discord.Interaction):
    model_names = [f for f in os.listdir("models") if os.path.isdir(os.path.join("models", f))]
    await interaction.response.send_message(f"Available models: {', '.join(model_names)}")

# Run the bot
bot.run(DISCORD_TOKEN)
