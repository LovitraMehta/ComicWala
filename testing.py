import torch
import numpy as np
from PIL import Image
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline

### --- STEP 1: Set Up the Streamlit UI --- ###

# Title and description
st.title("🖼️ Comic Strip Generator")
st.markdown("""
This app allows you to generate a comic strip by specifying a topic, the number of panels, and the art style.
""")

### --- STEP 2: Load TinyLlama for Text Generation --- ###
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to CPU
device = torch.device('cpu')
model = model.to(device)

# Initialize text generation pipeline on CPU
comic_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # Use CPU
)

### --- STEP 3: Load Stable Diffusion XL for High-Quality Images on CPU --- ###
model_id = "stabilityai/sd-turbo"  # Best for artistic comic style
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Move the pipeline to CPU
pipe.to('cpu')

### --- STEP 4: User Inputs a Prompt & Number of Panels --- ###
user_prompt = st.text_input("Enter a topic for the comic strip:", "Government of India")

# Get number of panels from the user
num_panels = st.slider("Choose the number of comic panels (3 to 6):", min_value=3, max_value=6, value=3)

### --- STEP 5: User Chooses an Art Style --- ###
art_styles = {
    "1": "Classic Comic",
    "2": "Anime",
    "3": "Cartoon",
    "4": "Noir",
    "5": "Cyberpunk",
    "6": "Watercolor"
}

art_choice = st.radio("Choose an Art Style for the Comic:", options=list(art_styles.values()))

### --- STEP 6: Generate Comic-Style Breakdown Using TinyLlama --- ###
instruction = (
    f"Generate a structured {num_panels}-panel comic strip description for the topic. "
    "Each panel should have a simple but clear scene description. "
    "Keep it short and focus on visuals for easy image generation.\n\n"
    "Topic: " + user_prompt + "\n\n"
    "Comic Strip Panels:\n"
)

response = comic_pipeline(
    instruction,
    max_new_tokens=400,  # Ensure full response
    temperature=0.7,
    repetition_penalty=1.1,
    do_sample=True
)[0]['generated_text']

# Extract only the structured comic description
comic_breakdown = response.replace(instruction, "").strip()
comic_panels = [line.strip() for line in comic_breakdown.split("\n") if line.strip()][:num_panels]

st.subheader("Comic Strip Breakdown:")
st.write("\n".join(comic_panels))  # Show generated panels

### --- STEP 7: Generate High-Quality Comic-Style Images on CPU --- ###
def generate_comic_image(description, style):
    """
    Generates a comic panel image using Stable Diffusion Turbo on CPU.
    """
    # Validate style input (fallback to "Comic" if invalid)
    valid_styles = ["Comic", "Anime", "Cyberpunk", "Watercolor", "Pixel Art"]
    chosen_style = style if style in valid_styles else "Comic"

    # Refined prompt (shorter, SD-Turbo-friendly)
    prompt = f"{description}, {chosen_style} style, bold outlines, vibrant colors, dynamic action."

    # Negative prompt (avoiding unwanted elements)
    negative_prompt = "blurry, distorted, text, watermark, low quality, extra limbs, messy background"

    try:
        # Generate image with optimized parameters
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,  # Faster processing
            guidance_scale=7
        ).images[0]
        return image
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return None  # Return None if generation fails

# Generate images for each panel
comic_images = [generate_comic_image(panel, art_choice) for panel in comic_panels]

# Remove None values if any images failed to generate
comic_images = [img for img in comic_images if img is not None]

if comic_images:
    ### --- STEP 8: Arrange Images in a Grid Based on Panel Count --- ###
    grid_map = {3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    rows, cols = grid_map.get(len(comic_images), (1, len(comic_images)))

    panel_width, panel_height = comic_images[0].size
    comic_strip = Image.new("RGB", (panel_width * cols, panel_height * rows))

    # Paste images in grid format
    for i, img in enumerate(comic_images):
        x_offset = (i % cols) * panel_width
        y_offset = (i // cols) * panel_height
        comic_strip.paste(img, (x_offset, y_offset))

    # Display and save the comic strip
    st.image(comic_strip, caption="Generated Comic Strip")
    comic_strip.save("comic_strip.png")
    st.success("✅ Comic strip saved as 'comic_strip.png'")
else:
    st.error("❌ No images were generated.")
