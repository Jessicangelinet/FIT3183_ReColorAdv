from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

class EnhanceAndResize:
    def __init__(self, contrast=1.1, sharpness=1.1, brightness=1.1):
        self.contrast = contrast
        self.sharpness = sharpness
        self.brightness = brightness

    def __call__(self, img):
        # Apply contrast enhancement
        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(self.contrast)

        # Apply sharpness enhancement
        enhancer_sharpness = ImageEnhance.Sharpness(img)
        img = enhancer_sharpness.enhance(self.sharpness)

        # Apply brightness enhancement
        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(self.brightness)

        # Resize the image to break down each pixel into 2x2 pixels
        width, height = img.size
        img = img.resize((width * 2, height * 2), Image.NEAREST)

        return img
