from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from numba import jit

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

def recolor_image_optimized(input_path, output_path, color_map):
    """
    Optimized recoloring function that uses:
    1. NumPy vectorized operations for common colors
    2. Numba-accelerated pixel-by-pixel replacement for rare colors
    """
    # Load the image
    image = Image.open(input_path).convert('RGB')

    # Convert image to numpy array
    img_array = np.array(image)

    # Create a new array for the output
    output_array = np.copy(img_array)

    # Split color mapping into common and rare colors
    common_colors = []
    rare_colors = []

    # Simple threshold - colors that make up less than 0.5% of the image are considered rare
    # This threshold can be adjusted based on your specific use case
    pixel_count = img_array.shape[0] * img_array.shape[1]

    # Analyze color frequencies
    colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
    color_frequency = {tuple(color): count/pixel_count for color, count in zip(colors, counts)}

    # Split the color map based on frequency
    for orig, new in color_map:
        orig_tuple = tuple(int(x) for x in orig)
        new_tuple = tuple(int(x) for x in new)

        if orig_tuple in color_frequency and color_frequency[orig_tuple] >= 0.005:  # 0.5%
            common_colors.append((orig_tuple, new_tuple))
        else:
            rare_colors.append((orig_tuple, new_tuple))

    # Process common colors with vectorized operations
    for orig, new in common_colors:
        # Create a boolean mask for each color
        r_mask = (img_array[:, :, 0] == orig[0])
        g_mask = (img_array[:, :, 1] == orig[1])
        b_mask = (img_array[:, :, 2] == orig[2])

        # Combined mask for exact color match
        mask = r_mask & g_mask & b_mask

        # Apply the new color where the mask is True
        output_array[mask, 0] = new[0]
        output_array[mask, 1] = new[1]
        output_array[mask, 2] = new[2]

    # Process rare colors with Numba for faster pixel-by-pixel operation
    if rare_colors:
        # Convert rare_colors to a format suitable for Numba
        rare_color_pairs = []
        for orig, new in rare_colors:
            rare_color_pairs.append((orig[0], orig[1], orig[2], new[0], new[1], new[2]))

        # Call the Numba-optimized function
        output_array = _process_rare_colors(output_array, np.array(rare_color_pairs))

    # Convert back to image
    new_image = Image.fromarray(output_array)

    # Save the recolored image
    new_image.save(output_path)
    print(f"Recolored image saved to {output_path}")

    return output_array

@jit(nopython=True)
def _process_rare_colors(image_array, rare_colors):
    """
    Numba-accelerated function to process rare colors pixel by pixel

    Parameters:
    image_array: NumPy array of the image
    rare_colors: Array of tuples (orig_r, orig_g, orig_b, new_r, new_g, new_b)

    Returns:
    Modified image array
    """
    height, width = image_array.shape[:2]

    # Process each pixel
    for y in range(height):
        for x in range(width):
            pixel = image_array[y, x]

            # Check against each rare color
            for color_pair in rare_colors:
                if (pixel[0] == color_pair[0] and
                    pixel[1] == color_pair[1] and
                    pixel[2] == color_pair[2]):
                    # Replace the color
                    image_array[y, x, 0] = color_pair[3]
                    image_array[y, x, 1] = color_pair[4]
                    image_array[y, x, 2] = color_pair[5]
                    break  # Stop checking once a match is found

    return image_array

def batch_process_images(input_paths, output_dir, color_map):
    """
    Process multiple images with the same color mapping

    Parameters:
    input_paths: List of paths to input images
    output_dir: Directory to save output images
    color_map: List of tuples ((r,g,b), (r,g,b)) for color mapping
    """
    import os
    from tqdm import tqdm

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image with progress bar
    for path in tqdm(input_paths, desc="Processing images"):
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, f"recolored_{filename}")
        recolor_image_optimized(path, output_path, color_map)

# Example usage
if __name__ == "__main__":
    input_image_path = 'input.png'
    output_image_path = 'output.png'

    # Color mapping from your original code
    color_mapping = [
        ((22, 194, 125), (255, 255, 255)),   # Medium sea green to White
        ((24, 195, 126), (255, 255, 255)),   # Medium sea green to White
        ((23, 60, 58), (0, 0, 0)),           # Dark green to Black
        ((25, 195, 126), (255, 255, 255)),  # Medium sea green to White
    ]

    # Single image processing
    recolor_image_optimized(input_image_path, output_image_path, color_mapping)
    get_image_colors(output_image_path)

    # For batch processing multiple images (uncomment to use)
    # import glob
    # input_images = glob.glob('images/*.png')
    # batch_process_images(input_images, 'output_images', color_mapping)

"""
Recolored image saved to output.png

Color counts in descending order:
186412: white (RGB: (np.uint8(255), np.uint8(255), np.uint8(255))) (76.95%)
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
216: medium sea green (RGB: (np.uint8(24), np.uint8(196), np.uint8(127))) (0.09%)
"""