#!/usr/bin/env python3
"""
Generate placeholder images for SafePath presentation
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Create img directory if needed
img_dir = '/home/ubuntu/.openclaw/workspace/safepath-proposal/img'
os.makedirs(img_dir, exist_ok=True)

# Image dimensions
WIDE_WIDTH = 1024
WIDE_HEIGHT = 576
SQUARE_SIZE = 512

def create_hero_image():
    """Create SafePath hero image"""
    img = Image.new('RGB', (WIDE_WIDTH, WIDE_HEIGHT), color=(20, 30, 48))
    draw = ImageDraw.Draw(img)

    # Draw path lines
    for y in range(WIDE_HEIGHT):
        if y % 20 < 10:
            draw.line([(WIDE_WIDTH//2, 0), (WIDE_WIDTH//2, WIDE_HEIGHT)], fill=(40, 60, 90), width=2)

    # Add SafePath text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    except:
        font = ImageFont.load_default()

    text = "SafePath"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (WIDE_WIDTH - text_width) // 2
    y = (WIDE_HEIGHT - text_height) // 2 - 50
    draw.text((x, y), text, fill=(100, 180, 255), font=font)

    # Subtitle
    try:
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except:
        sub_font = ImageFont.load_default()
    subtitle = "AI-Powered Scene Analysis"
    sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
    sub_width = sub_bbox[2] - sub_bbox[0]
    sub_x = (WIDE_WIDTH - sub_width) // 2
    draw.text((sub_x, y + 80), subtitle, fill=(150, 200, 255), font=sub_font)

    img.save(f'{img_dir}/safepath-hero.jpg', quality=95)
    print("Created: safepath-hero.jpg")

def create_cover_art_1():
    """Cover art - Blue professional theme"""
    img = Image.new('RGB', (SQUARE_SIZE, SQUARE_SIZE), color=(15, 25, 40))
    draw = ImageDraw.Draw(img)

    # Draw grid pattern
    for i in range(0, SQUARE_SIZE, 40):
        draw.line([(i, 0), (i, SQUARE_SIZE)], fill=(30, 50, 80), width=1)
        draw.line([(0, i), (SQUARE_SIZE, i)], fill=(30, 50, 80), width=1)

    # Draw circle with eye
    center = (SQUARE_SIZE//2, SQUARE_SIZE//2)
    radius = 100
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],
                 outline=(100, 180, 255), width=4)
    draw.ellipse([center[0]-30, center[1]-30, center[0]+30, center[1]+30],
                 fill=(100, 180, 255))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        font = ImageFont.load_default()

    text = "SafePath"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = (SQUARE_SIZE - text_width) // 2
    draw.text((x, 420), text, fill=(255, 255, 255), font=font)

    img.save(f'{img_dir}/cover-art-1.jpg', quality=95)
    print("Created: cover-art-1.jpg")

def create_cover_art_2():
    """Cover art - Yellow accent theme (nano banana style)"""
    img = Image.new('RGB', (SQUARE_SIZE, SQUARE_SIZE), color=(25, 20, 15))
    draw = ImageDraw.Draw(img)

    # Draw geometric shapes
    draw.rectangle([100, 100, 412, 412], outline=(255, 220, 0), width=5)
    draw.rectangle([150, 150, 362, 362], fill=(255, 220, 0), outline=(255, 200, 0), width=3)

    # Draw hazard icons
    draw.ellipse([50, 50, 150, 150], fill=(200, 50, 50), outline=(255, 100, 100), width=2)
    draw.ellipse([362, 50, 462, 150], fill=(200, 50, 50), outline=(255, 100, 100), width=2)
    draw.ellipse([206, 362, 306, 462], fill=(200, 50, 50), outline=(255, 100, 100), width=2)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
    except:
        font = ImageFont.load_default()

    text = "SafePath"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = (SQUARE_SIZE - text_width) // 2
    draw.text((x, 420), text, fill=(255, 220, 0), font=font)

    img.save(f'{img_dir}/cover-art-2.jpg', quality=95)
    print("Created: cover-art-2.jpg")

def create_hazard_diagram():
    """Create hazard detection diagram"""
    img = Image.new('RGB', (WIDE_WIDTH, WIDE_HEIGHT), color=(25, 25, 30))
    draw = ImageDraw.Draw(img)

    # Draw ground line
    draw.line([(0, WIDE_HEIGHT-100), (WIDE_WIDTH, WIDE_HEIGHT-100)],
              fill=(100, 100, 120), width=4)

    # Draw person icon (simple)
    person_x, person_y = WIDE_WIDTH//3, WIDE_HEIGHT-180
    draw.ellipse([person_x-30, person_y-80, person_x+30, person_y], fill=(100, 180, 255))
    draw.rectangle([person_x-20, person_y, person_x+20, person_y+80], fill=(80, 80, 100))

    # Draw hazards (potholes, obstacles)
    hazard1_x = WIDE_WIDTH//2 + 50
    draw.ellipse([hazard1_x-40, WIDE_HEIGHT-120, hazard1_x+40, WIDE_HEIGHT-60],
                 fill=(200, 60, 60), outline=(255, 100, 100), width=3)
    draw.ellipse([hazard1_x-25, WIDE_HEIGHT-105, hazard1_x+25, WIDE_HEIGHT-75],
                 fill=(20, 20, 25))

    # Draw detection box
    draw.rectangle([hazard1_x-50, WIDE_HEIGHT-140, hazard1_x+50, WIDE_HEIGHT-50],
                  outline=(0, 255, 150), width=4)
    draw.text((hazard1_x-20, WIDE_HEIGHT-130), "HAZARD!", fill=(0, 255, 150),
              font=ImageFont.load_default())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((50, 50), "Hazard Detection Concept", fill=(200, 200, 220), font=font)

    img.save(f'{img_dir}/hazard-detection.jpg', quality=95)
    print("Created: hazard-detection.jpg")

def create_pipeline_diagram():
    """Create pipeline flow diagram"""
    img = Image.new('RGB', (WIDE_WIDTH, WIDE_HEIGHT), color=(20, 25, 35))
    draw = ImageDraw.Draw(img)

    # Define boxes
    boxes = [
        ("Camera\nFeed", (100, 200, 250, 320), (100, 180, 255)),
        ("Pre-\nprocessing", (350, 200, 500, 320), (100, 180, 255)),
        ("DeepLabV3+\nModel", (600, 200, 750, 320), (100, 180, 255)),
        ("Segmentation\nOutput", (850, 200, 950, 320), (100, 180, 255)),
        ("PDF Report", (350, 400, 500, 520), (255, 180, 100))
    ]

    # Draw connecting arrows
    for i in range(len(boxes)-1):
        start_x = boxes[i][1][2]
        start_y = boxes[i][1][1] + 60
        end_x = boxes[i+1][1][0]
        end_y = boxes[i+1][1][1] + 60
        draw.line([(start_x, start_y), (end_x, end_y)], fill=(150, 200, 255), width=3)
        # Arrow head
        draw.polygon([(end_x, end_y), (end_x-10, end_y-10), (end_x-10, end_y+10)],
                     fill=(150, 200, 255))

    # Draw special arrow for report
    draw.line([(boxes[2][1][0] + 75, boxes[2][1][3]),
               (boxes[4][1][0] + 75, boxes[4][1][1])],
              fill=(255, 180, 100), width=3)

    # Draw boxes
    for label, box, color in boxes:
        draw.rectangle(box, outline=color, width=4)
        # Draw label
        try:
            box_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            box_font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=box_font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        x = box[0] + (box[2] - box[0] - label_width) // 2
        y = box[1] + (box[3] - box[1] - label_height) // 2
        draw.text((x, y), label, fill=color, font=box_font)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 35)
    except:
        title_font = ImageFont.load_default()

    draw.text((50, 50), "Processing Pipeline", fill=(200, 200, 220), font=title_font)

    img.save(f'{img_dir}/pipeline-diagram.jpg', quality=95)
    print("Created: pipeline-diagram.jpg")

def create_semantic_segmentation():
    """Create semantic segmentation illustration"""
    img = Image.new('RGB', (WIDE_WIDTH, WIDE_HEIGHT), color=(15, 20, 25))
    draw = ImageDraw.Draw(img)

    # Draw road
    draw.rectangle([0, WIDE_HEIGHT-200, WIDE_WIDTH, WIDE_HEIGHT], fill=(60, 70, 80))

    # Draw segmented areas (colored regions)
    # Sky
    draw.rectangle([0, 0, WIDE_WIDTH, WIDE_HEIGHT-200], fill=(40, 60, 90))

    # Buildings (left)
    draw.rectangle([0, 50, 300, WIDE_HEIGHT-200], fill=(100, 90, 70))
    draw.rectangle([20, 80, 280, WIDE_HEIGHT-230], fill=(120, 110, 90))

    # Buildings (right)
    draw.rectangle([700, 50, WIDE_WIDTH, WIDE_HEIGHT-200], fill=(100, 90, 70))
    draw.rectangle([720, 80, WIDE_WIDTH-20, WIDE_HEIGHT-230], fill=(120, 110, 90))

    # Road markings
    draw.line([(0, WIDE_HEIGHT-150), (WIDE_WIDTH, WIDE_HEIGHT-150)],
              fill=(255, 255, 200), width=8)
    draw.line([(0, WIDE_HEIGHT-70), (WIDE_WIDTH, WIDE_HEIGHT-70)],
              fill=(255, 255, 200), width=8)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 35)
    except:
        font = ImageFont.load_default()

    draw.text((50, 50), "Semantic Segmentation Output", fill=(200, 200, 220), font=font)

    # Legend
    legend_items = [
        ("Sky/Background", (40, 60, 90)),
        ("Buildings", (100, 90, 70)),
        ("Road/Safe", (60, 70, 80))
    ]
    y_offset = WIDE_HEIGHT - 180
    for label, color in legend_items:
        draw.rectangle([700, y_offset, 730, y_offset+30], fill=color, outline=(200, 200, 200), width=2)
        draw.text((740, y_offset+5), label, fill=(200, 200, 220), font=ImageFont.load_default())
        y_offset += 40

    img.save(f'{img_dir}/semantic-segmentation.jpg', quality=95)
    print("Created: semantic-segmentation.jpg")

if __name__ == '__main__':
    print("Generating images for SafePath presentation...")
    create_hero_image()
    create_cover_art_1()
    create_cover_art_2()
    create_hazard_diagram()
    create_pipeline_diagram()
    create_semantic_segmentation()
    print("\nAll images generated successfully!")
    print(f"Images saved to: {img_dir}")
