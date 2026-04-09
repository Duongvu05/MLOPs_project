import os
import io
import sys
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path for PyTorch imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
from torchvision import transforms
from training.pathology_training_utils import create_model_with_verification

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("uvicorn.error")

ml_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_NAMES = ["disc_herniation", "disc_bulging", "spondylolisthesis", "disc_narrowing"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load PyTorch Weights
    weights_path = project_root / "outputs/pathology_model/runs/efficientnet_b0/weights/best_model.pth"
    if weights_path.exists():
        logger.info(f"Loading weights from {weights_path} onto {device}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model_config = checkpoint.get("model_config", {})
        hyperparams = checkpoint.get("hyperparams", {})
        state_dict = checkpoint.get("model_state_dict", {})
        
        if "architecture" not in model_config:
            logger.warning("No architecture in model_config, guessing 'simple_multi_sequence_fusion'")
            model_config["architecture"] = "simple_multi_sequence_fusion"
            
        # Instantiate model structure
        model = create_model_with_verification(model_config, device, hyperparams=hyperparams, enable_wandb_logging=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        
        ml_models["model"] = model
        ml_models["model_config"] = model_config
        ml_models["hyperparams"] = hyperparams
        
        # Build normalizer using dataset configuration format
        data_config = hyperparams.get("data", {"input_size": [224, 224], "normalization": "imagenet"})
        input_size = tuple(data_config.get("input_size", [224, 224]))
        
        # Base augmentations pipeline for evaluate mode
        transform_list = [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
        normalization = data_config.get("normalization", "imagenet")
        if normalization == "imagenet":
            if model_config.get("in_channels", 1) == 1:
                transform_list.append(transforms.Normalize(mean=[0.449], std=[0.226]))
            else:
                transform_list.append(transforms.Normalize(mean=[0.485, 0.406, 0.456], std=[0.229, 0.224, 0.225]))
        elif normalization == "grayscale":
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
            
        ml_models["transform"] = transforms.Compose(transform_list)
        logger.info("✅ Pytorch model loaded successfully into memory.")
    else:
        logger.error(f"❌ Weights not found at {weights_path} — Make sure the model is trained!")
        
    yield
    ml_models.clear()

app = FastAPI(
    title="MRI Phenotyping API",
    description="API for multi-label classification of spinal pathology from MRI DICOM images.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Giao diện Web UI đơn giản dành cho bác sĩ để Upload ảnh.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Hệ thống AI Chẩn đoán MRI</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f4f8; margin: 0; padding: 40px; }
            .container { max-width: 650px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 5px; }
            h3 { color: #34495e; margin-bottom: 20px; text-align: center; font-weight: normal; font-size: 16px;}
            .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 30px; text-align: center; background: #fafbfc; margin-bottom: 25px; }
            input[type=file] { margin-top: 15px; font-size: 15px; }
            button { width: 100%; background: #2980b9; color: white; border: none; padding: 14px; font-size: 18px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: background 0.3s; }
            button:hover { background: #1a5276; }
            #result { margin-top: 25px; padding: 20px; background: #e8f8f5; border-left: 5px solid #1abc9c; border-radius: 4px; display: none; }
            #loading { display: none; text-align: center; margin-top: 20px; font-style: italic; color: #7f8c8d; }
            ul { list-style-type: none; padding-left: 0; }
            li { background: #ffffff; margin-bottom: 8px; padding: 10px 15px; border-radius: 4px; border: 1px solid #d1dbd9; display: flex; justify-content: space-between; }
            .high-risk { color: #c0392b; font-weight: bold;}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chẩn Đoán Bệnh Lý Cột Sống</h1>
            <h3>Chọn tối đa 4 lát cắt ảnh MRI (DICOM/PNG/JPG)</h3>
            
            <form id="uploadForm">
                <div class="upload-area">
                    <label style="font-weight: bold; color: #7f8c8d;">Kéo thả hoặc chọn tệp MRI tại đây</label>
                    <br>
                    <input type="file" id="files" name="files" multiple accept=".dcm,.jpg,.jpeg,.png">
                </div>
                
                <div style="margin-bottom: 25px; text-align: left;">
                    <label for="ivdLevel" style="font-weight: bold; color: #34495e;">📍 Chọn vị trí đĩa đệm (IVD Level):</label>
                    <select id="ivdLevel" name="ivdLevel" style="width: 100%; padding: 12px; margin-top: 8px; border-radius: 6px; border: 1px solid #bdc3c7; font-size: 16px;">
                        <option value="0">L1-L2</option>
                        <option value="1">L2-L3</option>
                        <option value="2">L3-L4</option>
                        <option value="3" selected>L4-L5</option>
                        <option value="4">L5-S1</option>
                    </select>
                </div>

                <button type="submit">Phân tích hình ảnh (AI Inference)</button>
            </form>
            
            <div id="loading">Đang tải Dữ liệu Model và Xử lý Ảnh. Xin hãy đợi...</div>
            <div id="result"></div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const files = document.getElementById('files').files;
                
                if (files.length === 0) {
                    alert("⚠️ Bác sĩ vui lòng chọn ít nhất 1 ảnh MRI!");
                    return;
                }
                if (files.length > 4) {
                    alert("⚠️ Hệ thống hiện chỉ hỗ trợ gộp tối đa 4 ảnh trong 1 lần phân tích!");
                    return;
                }

                const ivdValue = document.getElementById('ivdLevel').value;

                const formData = new FormData();
                for(let i=0; i<files.length; i++) {
                    formData.append('files', files[i]);
                }
                formData.append('ivd_level', parseInt(ivdValue));
                
                document.getElementById('result').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result').style.display = 'block';
                    
                    if (data.status === 'success') {
                        let html = "<h3 style='margin-top:0; text-align:left; color:#16a085;'>KẾT QUẢ DỰ ĐOÁN:</h3>";
                        html += "<ul>";
                        const mapping = {
                            "disc_herniation": "Thoát vị đĩa đệm",
                            "disc_bulging": "Phình đĩa đệm",
                            "spondylolisthesis": "Trượt đốt sống",
                            "disc_narrowing": "Hẹp khe khớp"
                        };
                        
                        for (const [key, value] of Object.entries(data.predictions)) {
                            const percent = (value * 100).toFixed(1);
                            const warningClass = value > 0.5 ? 'high-risk' : '';
                            html += `<li><span>${mapping[key] || key}:</span> <span class="${warningClass}">${percent}%</span></li>`;
                        }
                        html += "</ul>";
                        
                        const ivdMap = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"];
                        const checkedIvd = ivdMap[ivdValue] || "L4-L5";

                        html += `<div style="margin-top:15px; font-size:13px; color:#7f8c8d;">Báo cáo tại tầng <b>${checkedIvd}</b>. Đã xử lý thành công ${data.filenames.length} file: ${data.filenames.join(", ")}</div>`;
                        document.getElementById('result').innerHTML = html;
                    } else {
                        document.getElementById('result').innerHTML = "Lỗi hệ thống: " + data.detail;
                    }
                } catch (err) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('result').innerHTML = "Lỗi kết nối máy chủ!";
                }
            };
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def decode_image_to_tensor(file_bytes, filename: str, transform):
    """ Helper to decode uploaded bytes to a preprocessed PyTorch Tensor"""
    try:
        if filename.lower().endswith('.dcm'):
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            pixel_array = ds.pixel_array.astype(np.float32)
            if pixel_array.max() > pixel_array.min():
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            else:
                pixel_array = np.zeros_like(pixel_array)
            # Convert PyDicom normalized array [0,1] to [0,255] PIL Image
            image_pil = Image.fromarray((pixel_array * 255).astype(np.uint8))
        else:
            image_pil = Image.open(io.BytesIO(file_bytes)).convert("L")
            
        # Apply standard augmentations (Resize & Normalize)
        image_tensor = transform(image_pil)
        
        # Ensure correct dimensionality
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
            
        return image_tensor
    except Exception as e:
        logger.error(f"Error decoding image {filename}: {e}")
        return None

@app.post("/predict")
async def predict_mri(
    files: List[UploadFile] = File(...),
    ivd_level: int = Form(0, description="IVD Level: 0=L1-L2, 1=L2-L3, 2=L3-L4, 3=L4-L5, 4=L5-S1")
):
    """
    Endpoint to receive up to 4 MRI files, feed them through EfficientNet B0, and return exact inference results.
    """
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Pytorch Model weights not loaded. Has training completed?")
        
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided in the request")
        
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 files allowed per Request")
    
    filenames = [f.filename for f in files]
    logger.info(f"Received {len(files)} files: {filenames}")
    
    try:
        hyperparams = ml_models["hyperparams"]
        model = ml_models["model"]
        
        # Determine sequences expected by the model structure (e.g. ['sag_t2', 'ax_t2', 'sag_stir'])
        if hasattr(model, 'sequences'):
            expected_sequences = [s.lower() for s in model.sequences]
        else:
            multi_seq_config = hyperparams.get("multi_sequence", {})
            expected_sequences = [s.lower() for s in multi_seq_config.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])]
            
        # Container to hold mapped Input Tensors
        sequences_dict = {seq: None for seq in expected_sequences}
        
        for f in files:
            file_bytes = await f.read()
            tensor = decode_image_to_tensor(file_bytes, f.filename, ml_models["transform"])
            if tensor is None:
                continue
                
            tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
            
            # Naive Regex matching to assign files to sequences based on filename
            fn = f.filename.lower()
            matched_seq = None
            if "t2" in fn and "sag" in fn: matched_seq = "sag_t2"
            elif "t2" in fn and "ax" in fn: matched_seq = "ax_t2"
            elif "stir" in fn: matched_seq = "sag_stir"
            elif "t1" in fn: matched_seq = "sag_t1"
            
            # If the user selected a recognizable file (and it belongs to expected), slot it there
            if matched_seq and matched_seq in expected_sequences and sequences_dict[matched_seq] is None:
                sequences_dict[matched_seq] = tensor
            else:
                # Slot into the first empty expected sequence fallback
                for seq in expected_sequences:
                    if sequences_dict[seq] is None:
                        sequences_dict[seq] = tensor
                        break
        
        # Build Availability masks (True if sequence mapped, else False placeholder tensor)
        availability_dict = {}
        batch_size = 1
        for seq in expected_sequences:
            if sequences_dict[seq] is not None:
                availability_dict[seq] = torch.ones(batch_size, device=device, dtype=torch.bool)
            else:
                availability_dict[seq] = torch.zeros(batch_size, device=device, dtype=torch.bool)
                
        # Forward pass! Ensure gradients are detached.
        with torch.no_grad():
            ivd_tensor = torch.tensor([ivd_level], device=device) # Positional encode supplied by UI
            
            # SimpleMultiSequenceFusion has a direct ordered signature vs Adaptive models
            from models.pathology_model import SimpleMultiSequenceFusion
            if isinstance(model, SimpleMultiSequenceFusion):
                output = model(sequences_dict, availability_dict, ivd_tensor)
            else:
                output = model(sequences_dict, ivd_tensor, availability_dict, return_attention_weights=False)
                
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        real_predictions = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        
        return JSONResponse({
            "status": "success",
            "filenames": filenames,
            "predictions": real_predictions,
            "message": "Pytorch Inference successful"
        })
    except Exception as e:
        logger.error(f"Error processing files {filenames}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during PyTorch prediction")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
