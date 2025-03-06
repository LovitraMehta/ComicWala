import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline
from IPython.display import display

# --- STEP 1: Load TinyLlama for Story-Driven Text Generation ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize text generation pipeline
comic_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# --- STEP 2: Load Stable Diffusion Turbo for Compact Comic Panels ---
model_id = "stabilityai/sd-turbo"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")  # Move to GPU

# --- STEP 3: User Inputs a Prompt & Number of Panels ---
user_prompt = input("Enter a topic for the comic strip: ")

while True:
    try:
        num_panels = int(input("Enter the number of comic panels (3 to 20): "))
        if 3 <= num_panels <= 20:
            break
        else:
            print("❌ Please enter a number between 3 and 20.")
    except ValueError:
        print("❌ Invalid input! Please enter a number between 3 and 20.")

# --- STEP 4: User Chooses an Art Style ---
art_styles = {
    "1": "Classic Comic",
    "2": "Anime",
    "3": "Cartoon",
    "4": "Noir",
    "5": "Cyberpunk",
    "6": "Watercolor"
}

print("\n🎨 Choose an Art Style for the Comic:")
for key, style in art_styles.items():
    print(f"{key}. {style}")

while True:
    art_choice = input("\nEnter the number for your preferred art style: ")
    if art_choice in art_styles:
        chosen_style = art_styles[art_choice]
        print(f"✅ You selected: {chosen_style}")
        break
    else:
        print("❌ Invalid choice! Please enter a valid number.")

# --- STEP 5: Generate a Story-Driven Comic Breakdown Using TinyLlama ---
instruction = (
    f"Create a {num_panels}-panel comic story about {user_prompt}. "
    "Ensure a clear structure: a beginning, middle, and end. "
    "Each panel should describe a visually striking scene with concise action.\n\n"
    "Comic Strip Panels:\n"
)

response = comic_pipeline(
    instruction,
    max_new_tokens=600,  # Allow longer generation
    temperature=0.7,
    repetition_penalty=1.1,
    do_sample=True
)[0]['generated_text']

# Extract only the structured comic description
comic_breakdown = response.replace(instruction, "").strip()
comic_panels = [line.strip() for line in comic_breakdown.split("\n") if line.strip()][:num_panels]

print("\n🔹 Comic Strip Breakdown:\n", "\n".join(comic_panels))

# --- ADDING DESCRIPTION FUNCTION ---
def add_image_description(image, description, position=(20, 20)):
    """
    Adds a description on top of the generated comic strip.
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)  # Smaller font for compact panels
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), description, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Adjust position for readability
    bubble_width, bubble_height = text_width + 20, text_height + 15
    bubble_x, bubble_y = position

    # Draw a background box for readability
    draw.rectangle([bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height], fill="white", outline="black")
    draw.text((bubble_x + 10, bubble_y + 5), description, font=font, fill="black")

    return image

# --- STEP 6: Generate Compact Comic-Style Images ---
def generate_comic_image(description, style, short_description):
    """
    Generates a comic panel image using Stable Diffusion Turbo and adds a description of the image.
    """
    valid_styles = ["Comic", "Anime", "Cyberpunk", "Watercolor", "Pixel Art"]
    chosen_style = style if style in valid_styles else "Comic"

    # Refined prompt
    prompt = f"{description}, {chosen_style} style, bold outlines, dynamic action, high contrast."

    # Negative prompt to avoid issues
    negative_prompt = "blurry, distorted, text, watermark, low quality, extra limbs"

    try:
        # Generate image with optimized parameters
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # Lower steps for speed
            guidance_scale=6  # Slightly lower guidance for efficiency
        ).images[0]

        # Reduce size for compact layout
        image = image.resize((256, 256))

        # Add description
        return add_image_description(image, short_description, position=(10, 10))
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return None

# Generate images for each panel
comic_images = []
for i, panel in enumerate(comic_panels):
    short_description = f"Panel {i+1}: {panel}"
    comic_images.append(generate_comic_image(panel, chosen_style, short_description))

# Remove None values if any failed to generate
comic_images = [img for img in comic_images if img is not None]

# --- STEP 7: Arrange Images in a Compact Grid ---
if comic_images:
    # Define grid dimensions
    max_cols = 5  # Maximum 5 panels per row for compact layout
    rows = (len(comic_images) // max_cols) + (1 if len(comic_images) % max_cols else 0)
    cols = min(len(comic_images), max_cols)

    panel_width, panel_height = comic_images[0].size
    comic_strip = Image.new("RGB", (panel_width * cols, panel_height * rows))

    # Paste images in grid format
    for i, img in enumerate(comic_images):
        x_offset = (i % cols) * panel_width
        y_offset = (i // cols) * panel_height
        comic_strip.paste(img, (x_offset, y_offset))

    # Display and save the compact comic strip
    display(comic_strip)
    comic_strip.save("compact_comic_strip.png")
    print("\n✅ Comic strip saved as 'compact_comic_strip.png'")
else:
    print("\n❌ No images were generated.")
