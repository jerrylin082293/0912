from inference_realesrgan import run_gan
import cv2

img = cv2.imread('input_image.png')

enhanced_image = run_gan(img)
cv2.imwrite('output_image.png', enhanced_image)

