from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Path to the input image
img_path = './imgs_en/mom.jpg'
slice = {'horizontal_stride': 1500, 'vertical_stride': 500, 'merge_x_thres': 50, 'merge_y_thres': 35}
results = ocr.ocr(img_path, cls=True, slice=slice)

# Load the input image
original_image = Image.open(img_path).convert("RGB")

# Create two images: one for bounding boxes and one for text only
image_with_boxes = original_image.copy()
image_with_text = Image.new("RGB", original_image.size, (255, 255, 255))  # Create a white image for text only

# Initialize drawing contexts
draw_boxes = ImageDraw.Draw(image_with_boxes)
draw_text = ImageDraw.Draw(image_with_text)
font = ImageFont.truetype(r'C:\Users\Aleena Ajith\OneDrive\Desktop\OCR\PaddleOCR\doc\fonts\simfang.ttf', size=12)  # Adjust size as needed

# Process and draw results
for res in results:
    for line in res:
        box = [tuple(point) for point in line[0]]
        txt = line[1][0]
        # Draw bounding box on the original image (without text)
        draw_boxes.polygon(box, outline="red", width=2)
        # Draw text only on the white image
        draw_text.text((box[0][0], box[0][1]), txt, fill="black", font=font)

# Save the two output images
image_with_boxes.save("result_with_boxes_only.png")  # Image with detected bounding boxes only
image_with_text.save("result_only_text.png")         # Image with only detected text on white background
