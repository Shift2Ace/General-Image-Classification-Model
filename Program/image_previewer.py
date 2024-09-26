from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = input("Image file path: ")  # Replace with your image path
original_image = Image.open(image_path)

# Define the sizes to resize to
sizes = [2**i for i in range(3, 12)]  # 8x8 to 2048x2048

# Create a figure to display the images in 3 columns and 3 rows
fig, axes = plt.subplots(3, 3, figsize=(5, 5))  # Adjusted window size

# Resize and display each image
for ax, size in zip(axes.flatten(), sizes):
    resized_image = original_image.resize((size, size))
    ax.imshow(resized_image)
    ax.set_title(f'{size}x{size}')
    ax.axis('off')

plt.tight_layout()
plt.show()

