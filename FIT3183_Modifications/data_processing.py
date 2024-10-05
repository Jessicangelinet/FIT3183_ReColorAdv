import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter

# Define the EnhanceAndResize class
class EnhanceAndResize:
    def __init__(self, contrast=0.1, sharpness=0.1, brightness=0.3):
        self.contrast = contrast
        self.sharpness = sharpness
        self.brightness = brightness

    def __call__(self, img):
        # Resize the image to break down each pixel into 2x2 pixels
        width, height = img.size
        img = img.resize((width * 2, height * 2), Image.NEAREST)

        # # Apply contrast enhancement
        # enhancer_contrast = ImageEnhance.Contrast(img)
        # img = enhancer_contrast.enhance(self.contrast)

        # Apply sharpness enhancement
        # enhancer_sharpness = ImageEnhance.Sharpness(img)
        # img = enhancer_sharpness.enhance(self.sharpness)

        # # Apply brightness enhancement
        # enhancer_brightness = ImageEnhance.Brightness(img)
        # img = enhancer_brightness.enhance(self.brightness)

        return img

class BlurAndResize:
    def __init__(self, blur_radius=0.2):
        self.blur_radius = blur_radius

    def __call__(self, img):
        # Apply blur effect
        img = img.filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Resize the image to break down each pixel into 2x2 pixels
        width, height = img.size
        img = img.resize((width * 2, height * 2), Image.NEAREST)

        return img

