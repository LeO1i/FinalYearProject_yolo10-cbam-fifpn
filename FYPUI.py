import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import zipfile
from datetime import datetime
import io

# Load our trained YOLOv10_CBAM model
model = YOLO(r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt")

# Store brain tumor classes into class_name
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def BrainTumorDetection(input_image):
    """
    Process a single image for brain tumor detection
    """
    # Convert the image from RGB (Gradio default) to BGR (OpenCV default)
    image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Run inference on the image by setting confidence threshold at 50%
    results = model.predict(source=image_bgr, conf=0.5)

    # Prepare a list for detections; each detection is (box, label, confidence)
    detections = []

    if results and len(results):
        # We are processing only one image at a time
        result = results[0]

        if result.boxes is not None:
            # Get bounding box coordinates, class indices, and confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()  # Coordinates: [x1, y1, x2, y2]
            class_ids = result.boxes.cls.cpu().numpy()  # Class indices
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores

            # Loop through a combination of 3 array
            for box, class_idz, conf in zip(boxes, class_ids, confs):
                label = class_names[int(class_idz)]     # Map class index into class name
                detections.append((box, label, conf))   # Store detected object info in detections list

    # Convert the original image to a PIL Image for drawing
    output_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.load_default()

    # Draw each detection as a rectangle with a label
    for (box, label, conf) in detections:
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Create a background for the text
        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((x1, max(0, y1 - 20)), text, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((x1, max(0, y1 - 20)), text, fill="white", font=font)

    # Return the output image as a NumPy array
    return np.array(output_image)

def process_batch_images(image_files):
    """
    Process multiple images for batch detection
    """
    processed_images = []
    detection_results = []
    
    for img_file in image_files:
        # Read image
        img = cv2.imread(img_file.name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        processed_img = BrainTumorDetection(img_rgb)
        processed_images.append(processed_img)
        
        # Get detection info for this image
        results = model.predict(source=img, conf=0.5)
        detections = []
        
        if results and len(results) and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, class_idz, conf in zip(boxes, class_ids, confs):
                label = class_names[int(class_idz)]
                detections.append(f"{label}: {conf:.2f}")
        
        detection_results.append(f"Image: {os.path.basename(img_file.name)} - Detections: {', '.join(detections) if detections else 'None'}")
    
    return processed_images, "\n".join(detection_results)

def create_download_zip(processed_images, original_filenames):
    """
    Create a zip file containing processed images for download without creating temp JPG files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"brain_tumor_detection_results_{timestamp}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for img, filename in zip(processed_images, original_filenames):
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=95)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            zip_name = f"processed_{base_name}.jpg"
            zipf.writestr(zip_name, buffer.getvalue())

    return zip_path

def single_image_interface(input_image):
    """
    Interface for single image processing
    """
    if input_image is None:
        return None, "Please upload an image first."
    
    processed_image = BrainTumorDetection(input_image)
    return processed_image, "Image processed successfully! You can download the result by right-clicking on the image."

def batch_interface(image_files):
    """
    Interface for batch image processing
    """
    if not image_files:
        return None, gr.update(visible=False, value=None), "Please upload images first."
    
    try:
        processed_images, detection_summary = process_batch_images(image_files)
        
        # Create download zip
        original_filenames = [f.name for f in image_files]
        zip_path = create_download_zip(processed_images, original_filenames)
        
        return processed_images, gr.update(value=zip_path, visible=True), detection_summary
    except Exception as e:
        return None, gr.update(visible=False, value=None), f"Error processing images: {str(e)}"

# Create the Gradio interface with tabs
with gr.Blocks(title="Brain Tumor Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Brain Tumor Detection System")
    gr.Markdown("Upload MRI images to detect brain tumors using our trained YOLOv10 model.")
    
    with gr.Tabs():
        # Single Image Tab
        with gr.Tab("Single Image Detection"):
            gr.Markdown("### Upload a single MRI image for tumor detection")
            
            with gr.Row():
                with gr.Column():
                    single_input = gr.Image(
                        label="Upload MRI Image",
                        type="numpy",
                        height=400
                    )
                    single_process_btn = gr.Button("Detect Tumor", variant="primary")
                
                with gr.Column():
                    single_output = gr.Image(
                        label="Detection Result",
                        height=400
                    )
                    single_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            single_process_btn.click(
                fn=single_image_interface,
                inputs=single_input,
                outputs=[single_output, single_status]
            )
        
        # Batch Processing Tab
        with gr.Tab("Batch Image Detection"):
            gr.Markdown("### Upload multiple MRI images for batch tumor detection")
            
            with gr.Row():
                with gr.Column():
                    batch_input = gr.File(
                        label="Upload Multiple Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    batch_process_btn = gr.Button("Process All Images", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Gallery(
                        label="Detection Results",
                        height=400,
                        columns=2,
                        rows=2
                    )
                    batch_download = gr.File(
                        label="Download Results (ZIP)",
                        visible=False
                    )
            
            batch_status = gr.Textbox(
                label="Processing Summary",
                interactive=False,
                lines=5
            )
            
            batch_process_btn.click(
                fn=batch_interface,
                inputs=batch_input,
                outputs=[batch_output, batch_download, batch_status]
            )
    
    # Information section
    with gr.Accordion("ℹ️ Information", open=False):
        gr.Markdown("""
        ### About the Model
        - **Model**: YOLOv10 with CBAM (Convolutional Block Attention Module)
        - **Classes**: Glioma, Meningioma, No Tumor, Pituitary
        - **Confidence Threshold**: 50%
        
        ### How to Use
        1. **Single Image**: Upload one MRI image and click "Detect Tumor"
        2. **Batch Processing**: Upload multiple images and click "Process All Images"
        3. **Download**: Right-click on results or use the ZIP download for batch results
        
        ### Detection Classes
        - **Glioma**: A type of tumor that occurs in the brain and spinal cord
        - **Meningioma**: A tumor that forms on membranes that cover the brain and spinal cord
        - **No Tumor**: Normal brain tissue without any tumor
        - **Pituitary**: A tumor in the pituitary gland
        """)

if __name__ == "__main__":
    demo.launch()





