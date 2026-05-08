import gradio as gr
import cv2
import torch
import numpy as np
import time

# Import your model architecture from your modularized file
from sod_model import SaliencyBaseline  # Update this to SaliencyImproved if you used the U-Net class!

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
print("🚀 Initializing Gradio Web Demo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the trained weights
model = SaliencyBaseline().to(device)

try:
    # Use map_location just in case someone tests this on a machine without a GPU
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("✅ Successfully loaded 'best_model.pth'")
except FileNotFoundError:
    print("⚠️ Could not find 'best_model.pth'. Please ensure the file is in the same directory.")
    exit()

model.eval() # Freeze layers for inference

# ==========================================
# 2. PREDICTION PIPELINE
# ==========================================
def predict_saliency(input_image):
    # 1. Start the inference timer
    start_time = time.time()
    
    # 2. Preprocess the image (OpenCV Requirement)
    img_resized = cv2.resize(input_image, (128, 128))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 3. Run the model
    with torch.no_grad():
        pred = model(img_tensor)
        
    # 4. Stop the timer
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    time_string = f"{inference_time_ms:.2f} ms"
        
    # 5. Post-process predictions for display
    mask_np = pred.squeeze().cpu().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 6. Create the OpenCV Overlay (Heatmap)
    heatmap = np.zeros_like(img_resized)
    heatmap[:, :, 0] = mask_bin 
    overlay = cv2.addWeighted(img_resized, 0.7, heatmap, 0.5, 0)
    
    # Format mask for Gradio (1-channel to 3-channel RGB)
    mask_display = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB)
    
    return img_resized, mask_display, overlay, time_string

# ==========================================
# 3. BUILD & LAUNCH UI
# ==========================================
demo = gr.Interface(
    fn=predict_saliency,
    inputs=gr.Image(label="Upload any Image"),
    outputs=[
        gr.Image(label="1. Input Image (128x128)"),
        gr.Image(label="2. Output Saliency Mask"),
        gr.Image(label="3. Overlayed Visualization"),
        gr.Textbox(label="4. Inference Time per Image")
    ],
    title="Subject Saliency Detector (Genpact Demo)",
    description="Upload an image to see the model extract the subject and calculate real-time inference speed.",
    theme="default"
)

# Only launch if the script is run directly
if __name__ == "__main__":
    demo.launch(share=False)