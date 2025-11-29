import numpy as np
from PIL import Image
import torch


class ImageAutosize:

	@classmethod
	def INPUT_TYPES(s):
		return {
		    "required": {
		        "image": ("IMAGE", ),
		        "max_size": ("INT", {
		            "default": 1280,
		            "min": 1,
		            "max": 8192,
		            "step": 1
		        }),
		        "min_size": ("INT", {
		            "default": 512,
		            "max": 4096,
		            "min": 1,
		            "step": 1
		        }),
		        "divisible_by": ("INT", {
		            "default": 32,
		            "min": 1
		        }),
		        "interpolation_mode": (["nearest", "lanczos", "bilinear", "bicubic", "box", "hamming"], {
		            "default": "lanczos",
		        }),
		        "crop_mode": (["none", "center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"], {
		            "default": "center",
		        })
		    }
		}

	RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
	RETURN_NAMES = ("image", "width", "height", "multiplier")
	FUNCTION = "op"
	CATEGORY = "image"
	DESCRIPTION = """Resizes the input image"""

	def get_crop_coords(self, img_width, img_height, target_width, target_height, crop_mode):
		if crop_mode == "center":
			left = (img_width - target_width) // 2
			top = (img_height - target_height) // 2
		elif crop_mode == "top":
			left = (img_width - target_width) // 2
			top = 0
		elif crop_mode == "bottom":
			left = (img_width - target_width) // 2
			top = img_height - target_height
		elif crop_mode == "left":
			left = 0
			top = (img_height - target_height) // 2
		elif crop_mode == "right":
			left = img_width - target_width
			top = (img_height - target_height) // 2
		elif crop_mode == "top_left":
			left = 0
			top = 0
		elif crop_mode == "top_right":
			left = img_width - target_width
			top = 0
		elif crop_mode == "bottom_left":
			left = 0
			top = img_height - target_height
		elif crop_mode == "bottom_right":
			left = img_width - target_width
			top = img_height - target_height
		else:  # fallback to center
			left = (img_width - target_width) // 2
			top = (img_height - target_height) // 2

		right = left + target_width
		bottom = top + target_height
		return left, top, right, bottom

	def op(self, image, max_size, min_size, divisible_by, interpolation_mode, crop_mode):
		total_images = image.shape[0]
		out_images = []
		interpolation_mode = getattr(Image, interpolation_mode.upper(), Image.LANCZOS)

		for i in range(total_images):
			img_array = 255. * image[i].cpu().numpy()
			width = img_array.shape[1]
			height = img_array.shape[0]

			larger_dimension = max(width, height)
			multiplier = max_size / larger_dimension
			width *= multiplier
			height *= multiplier

			smaller_dimension = min(width, height)
			if (smaller_dimension < min_size):
				multiplier = min_size / smaller_dimension
				width *= multiplier
				height *= multiplier

			width = int(round(width / divisible_by) * divisible_by)
			height = int(round(height / divisible_by) * divisible_by)

			final_multiplier = width / img_array.shape[1]

			img = Image.fromarray(img_array.astype(np.uint8))

			if crop_mode != "none":
				scale = max(width / img.width, height / img.height)
				new_size = (int(round(img.width * scale)), int(round(img.height * scale)))
				img = img.resize(new_size, interpolation_mode)

				left, top, right, bottom = self.get_crop_coords(img.width, img.height, width, height, crop_mode)
				img = img.crop((left, top, right, bottom))
			else:
				img = img.resize((width, height), interpolation_mode)

			img_array = np.clip(np.array(img), 0, 255).astype(np.uint8)
			out_images.append(img_array)

		restored_img_np = np.array(out_images).astype(np.float32) / 255.0
		restored_img_tensor = torch.from_numpy(restored_img_np)

		return (
		    restored_img_tensor,
		    int(width),
		    int(height),
		    float(final_multiplier),
		)


NODE_CLASS_MAPPINGS = {
    "ImageAutosize": ImageAutosize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAutosize": "Image Autosize",
}
