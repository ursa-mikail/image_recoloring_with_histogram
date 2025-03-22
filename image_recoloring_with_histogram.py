from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math

# Enhanced color name dictionary with better matching
def rgb_to_color_name(rgb):
    # Convert numpy uint8 to regular integers if needed
    rgb = tuple(int(c) for c in rgb)

    # Base colors for reference
    color_dict = {
        (255, 0, 0): 'red',
        (0, 255, 0): 'green',
        (0, 0, 255): 'blue',
        (255, 255, 0): 'yellow',
        (0, 0, 0): 'black',
        (255, 255, 255): 'white',
        (255, 165, 0): 'orange',
        (128, 0, 128): 'purple',
        (0, 255, 255): 'cyan',
        (255, 192, 203): 'pink',
        (165, 42, 42): 'brown',
        (128, 128, 128): 'gray',
        (0, 128, 0): 'dark green',
        (0, 100, 0): 'dark green',
        (34, 139, 34): 'forest green',
        (50, 205, 50): 'lime green',
        (144, 238, 144): 'light green',
        (152, 251, 152): 'pale green',
        (143, 188, 143): 'dark sea green',
        (60, 179, 113): 'medium sea green',
        (46, 139, 87): 'sea green',
        (32, 178, 170): 'light sea green',
        (0, 139, 139): 'dark cyan',
        (0, 206, 209): 'dark turquoise',
        (64, 224, 208): 'turquoise',
        (72, 209, 204): 'medium turquoise',
        (175, 238, 238): 'pale turquoise'
    }

    # Exact match
    if rgb in color_dict:
        return color_dict[rgb]

    # Find the closest color by Euclidean distance
    min_distance = float('inf')
    closest_color = None

    for known_rgb, color_name in color_dict.items():
        # Calculate Euclidean distance
        distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(rgb, known_rgb)))

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    # If it's reasonably close to a known color
    if min_distance < 50:
        intensity = sum(float(c) for c in rgb) / 3  # Average intensity

        # Adjust with intensity descriptors
        if intensity > 200:
            prefix = "light "
        elif intensity < 100:
            prefix = "dark "
        else:
            prefix = ""

        return f"{prefix}{closest_color}"

    # Describe by primary components
    r, g, b = [float(c) for c in rgb]

    # Special case for grayscale
    if abs(r - g) < 20 and abs(r - b) < 20:
        if r > 200:
            return "light gray"
        elif r > 100:
            return "gray"
        else:
            return "dark gray"

    # Initialize base to a default value
    base = "unknown"

    # Determine dominant color
    if r > g and r > b:
        base = "red"
        if g > 100:
            base = "orange" if g < 200 else "yellow"
    elif g > r and g > b:
        base = "green"
        if b > 100:
            base = "teal" if b < 200 else "cyan"
    elif b > r and b > g:
        base = "blue"
        if r > 100:
            base = "purple" if r < 200 else "magenta"
    # If none of the above conditions is true (e.g., equal values)
    else:
        # Additional grayscale check
        if abs(r - g) < 10 and abs(r - b) < 10:
            if r > 200:
                base = "light gray"
            elif r > 100:
                base = "gray"
            else:
                base = "dark gray"

    # Intensity modifier
    if (r + g + b) / 3 > 180:
        base = f"light {base}"
    elif (r + g + b) / 3 < 80:
        base = f"dark {base}"

    return base

def get_image_colors(input_path):
    # Load the image
    image = Image.open(input_path).convert('RGB')

    # Convert image to numpy array
    img_array = np.array(image)

    # Flatten the array to a list of RGB tuples
    pixels = [tuple(pixel) for pixel in img_array.reshape(-1, 3)]

    # Count the frequency of each color
    color_counts = Counter(pixels)

    # Sort by frequency
    most_common_colors = color_counts.most_common(20)  # Show top 20 colors

    # Display the counts in descending order
    print("\nColor counts in descending order:")
    for i, (color, count) in enumerate(most_common_colors):
        percentage = (count / len(pixels)) * 100
        color_name = rgb_to_color_name(color)
        print(f"{count}: {color_name} (RGB: {color}) ({percentage:.2f}%)")

    # Plot color histogram with descriptions
    plt.figure(figsize=(14, 8))
    colors = [np.array([int(c) for c in color]) / 255 for color, _ in most_common_colors]
    counts = [count for _, count in most_common_colors]

    # Create horizontal bar chart for better readability
    y_pos = range(len(most_common_colors))
    plt.barh(y_pos, counts, color=colors)

    # Add labels with counts
    for i, (count, color) in enumerate(zip(counts, most_common_colors)):
        percentage = (count / len(pixels)) * 100
        color_name = rgb_to_color_name(color[0])
        rgb_values = color[0]
        plt.text(count + (max(counts) * 0.01), i,
                 f"{color_name} (RGB: {rgb_values}) - {count} ({percentage:.2f}%)",
                 va='center')

    plt.title('Color Distribution in Image')
    plt.xlabel('Pixel Count')
    plt.ylabel('Colors')
    plt.tight_layout()
    plt.show()

def recolor_image(input_path, output_path, color_map):
    # Load the image
    image = Image.open(input_path).convert('RGB')

    # Convert image to numpy array
    img_array = np.array(image)

    # Create a mapping dictionary for color replacement
    color_map_dict = {}
    for orig, new in color_map:
        # Convert to tuple of ints for dictionary keys
        orig_tuple = tuple(int(x) for x in orig)
        new_tuple = tuple(int(x) for x in new)
        color_map_dict[orig_tuple] = new_tuple

    # Create output array
    output_array = np.copy(img_array)

    # Apply color mapping
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            pixel = tuple(img_array[y, x])
            if pixel in color_map_dict:
                output_array[y, x] = color_map_dict[pixel]

    # Convert back to image
    new_image = Image.fromarray(output_array)

    # Save the recolored image
    new_image.save(output_path)
    print(f"Recolored image saved to {output_path}")

# Example usage
input_image_path = 'input.png'
output_image_path = 'output.png'

# Color mapping from your original code
color_mapping = [
    ((22, 194, 125), (255, 255, 255)),   # Medium sea green to White
    ((24, 195, 126), (255, 255, 255)),   # Medium sea green to White
    ((23, 60, 58), (0, 0, 0)),           # Dark green to Black
]

recolor_image(input_image_path, output_image_path, color_mapping)
get_image_colors(output_image_path)

"""
Recolored image saved to output.png

Color counts in descending order:
181645: medium sea green (RGB: (np.uint8(25), np.uint8(195), np.uint8(126))) (74.98%)
4767: white (RGB: (np.uint8(255), np.uint8(255), np.uint8(255))) (1.97%)
2661: black (RGB: (np.uint8(0), np.uint8(0), np.uint8(0))) (1.10%)
2318: medium sea green (RGB: (np.uint8(23), np.uint8(194), np.uint8(125))) (0.96%)
791: medium sea green (RGB: (np.uint8(21), np.uint8(194), np.uint8(125))) (0.33%)
630: medium sea green (RGB: (np.uint8(24), np.uint8(194), np.uint8(125))) (0.26%)
532: light white (RGB: (np.uint8(254), np.uint8(254), np.uint8(254))) (0.22%)
488: dark green (RGB: (np.uint8(22), np.uint8(59), np.uint8(57))) (0.20%)
380: medium sea green (RGB: (np.uint8(24), np.uint8(194), np.uint8(126))) (0.16%)
373: medium sea green (RGB: (np.uint8(25), np.uint8(194), np.uint8(126))) (0.15%)
342: light white (RGB: (np.uint8(253), np.uint8(253), np.uint8(253))) (0.14%)
335: medium sea green (RGB: (np.uint8(21), np.uint8(196), np.uint8(126))) (0.14%)
317: medium sea green (RGB: (np.uint8(23), np.uint8(195), np.uint8(126))) (0.13%)
285: medium sea green (RGB: (np.uint8(20), np.uint8(194), np.uint8(125))) (0.12%)
279: medium sea green (RGB: (np.uint8(20), np.uint8(193), np.uint8(124))) (0.12%)
261: medium sea green (RGB: (np.uint8(20), np.uint8(195), np.uint8(126))) (0.11%)
256: dark unknown (RGB: (np.uint8(22), np.uint8(55), np.uint8(55))) (0.11%)
253: medium sea green (RGB: (np.uint8(19), np.uint8(194), np.uint8(125))) (0.10%)
248: medium sea green (RGB: (np.uint8(21), np.uint8(195), np.uint8(126))) (0.10%)
219: medium sea green (RGB: (np.uint8(22), np.uint8(195), np.uint8(126))) (0.09%)
"""