from datetime import datetime
from math import ceil, sqrt

from PIL import Image

from src.util import get_workflow


def create_gif_collage(images, image_workflow):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    collage_path = f"./out/images_{timestamp}"
    exif_info = images[0].getexif()
    images[0].save(collage_path, 'webp', save_all=True, append_images=images[1:], loop=0, allow_mixed=True, exif=exif_info)

    return collage_path


def create_collage(images, image_workflow = None):
    if images is None or len(images) == 0:
        print("Error: No images to make collage")
        return None

    if images[0].format == 'WEBP' or 'GIF' or 'MP4':
        return create_gif_collage(images, image_workflow)

    num_images = len(images)
    num_cols = ceil(sqrt(num_images))
    num_rows = ceil(num_images / num_cols)
    collage_width = max(image.width for image in images) * num_cols
    collage_height = max(image.height for image in images) * num_rows
    collage = Image.new('RGB', (collage_width, collage_height))

    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        x_offset = col * image.width
        y_offset = row * image.height
        collage.paste(image, (x_offset, y_offset))

    pnginfo = get_workflow(images[0], image_workflow)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    collage_path = f"./out/images_{timestamp}.png"
    collage.save(collage_path, pnginfo=pnginfo)

    return collage_path
