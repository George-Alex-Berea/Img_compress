import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_color_image(path):
    """Load image and return as 3D NumPy array (H x W x 3)."""
    image = Image.open(path).convert('RGB')
    mat_img = np.array(image)
    size = mat_img.shape
    k = min(size[0], size[1]) // 40  # Set k to be 1/40th of the smaller dimension, modify at will
    return (mat_img, k)

def svd_compress_color(image_array, k):
    """Apply SVD compression separately to each RGB channel."""
    compressed_channels = []
    for channel in range(3):  # R, G, B
        A = image_array[:, :, channel]
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        compressed_channel = np.dot(U_k, np.dot(S_k, Vt_k))
        compressed_channels.append(np.clip(compressed_channel, 0, 255))
    return np.stack(compressed_channels, axis=2).astype('uint8')

def save_image(matrix, output_path):
    Image.fromarray(matrix).save(output_path)

image_path = 'image_name'        # Replace with your input file
output_path = 'output_name'

(original ,k) = load_color_image(image_path)
compressed = svd_compress_color(original, k)
save_image(compressed, output_path)
