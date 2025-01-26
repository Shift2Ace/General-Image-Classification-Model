import os
import random
from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_path, text, output_path):
    # Open an image file
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        width, height = img.size
        
        # Create a new image for the text with transparent background
        text_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Load a bold font
        font_size = 1  # Start with a small font size
        font = ImageFont.truetype("arialbd.ttf", font_size)
        
        # Increase font size until the text fits the image width
        while True:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            if text_width >= width or text_height >= height:
                break
            font_size += 1
            font = ImageFont.truetype("arialbd.ttf", font_size)
        
        # Calculate text position
        text_x = (width - text_width) / 2
        
        # Add repeated text to the text image from top to bottom with 50% opacity
        current_y = -500
        while current_y < height * 1.1:
            # Draw outline
            outline_range = 2  # Thickness of the outline
            for x_offset in range(-outline_range, outline_range + 1):
                for y_offset in range(-outline_range, outline_range + 1):
                    if x_offset != 0 or y_offset != 0:
                        draw.text((text_x + x_offset, current_y + y_offset), text, font=font, fill=(0, 0, 0, 128))  # Black outline with 50% opacity
            # Draw main text
            draw.text((text_x, current_y), text, font=font, fill=(255, 255, 255, 128))  # 50% opacity
            current_y += text_height
        
        # Combine the original image with the text image
        combined = Image.alpha_composite(img, text_img)
        
        # Save the image with the added text
        combined.save(output_path)

# Input text and repeat until correct length is entered
while True:
    text = input("Enter a text (3 to 20 letters): ")
    if 3 <= len(text) <= 20:
        break
    else:
        print("The input text must be between 3 to 20 letters. Please try again.")

# Input folder path of valid dataset that has different folders with multiple images
dataset_path = input("Enter the dataset folder path: ")

# Input number of output images
num_output_images = int(input("Enter the number of output images: "))

# Input position to save the new folder
save_position = input("Enter the position to save the new folder: ")

# Input new folder name
new_folder_name = input("Enter the new folder name: ")

# Create a new folder in the chosen position with the given name
new_folder_path = os.path.join(save_position, new_folder_name)
os.makedirs(new_folder_path, exist_ok=True)

# Get a list of all image files in the dataset folder and its subfolders
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Ensure that the number of output images does not exceed the number of available images
num_output_images = min(num_output_images, len(image_files))

# Randomly select the specified number of unique images from the dataset
selected_images = random.sample(image_files, num_output_images)

# Add text to each selected image and save them in the new folder
for i, image_file in enumerate(selected_images):
    output_path = os.path.join(new_folder_path, f"image_{i+1}.png")
    add_text_to_image(image_file, text, output_path)

print(f"Processed {num_output_images} images and saved them in {new_folder_path}")