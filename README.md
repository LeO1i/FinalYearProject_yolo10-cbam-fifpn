# 🧠 Brain Tumor Detection System

A comprehensive web-based interface for detecting brain tumors in MRI images using YOLOv10 with CBAM (Convolutional Block Attention Module).

## 🚀 Features

- **Single Image Detection**: Upload and process individual MRI images
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Results**: View detection results with bounding boxes and confidence scores
- **Download Functionality**: Download processed images individually or as a ZIP file
- **User-friendly Interface**: Modern, responsive web interface built with Gradio
- **Multiple Tumor Types**: Detects Glioma, Meningioma, No Tumor, and Pituitary tumors

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file exists**:
   Ensure the trained model file is located at:
   ```
   Trained_model/YOLOv10CM_FYPtrained.pt
   ```

## 🚀 Quick Start

### Option 1: Using the launcher script (Recommended)
```bash
python run_ui.py
```

### Option 2: Direct execution
```bash
python FYPUI.py
```

### Option 3: Using Gradio directly
```bash
gradio FYPUI.py
```

## 🌐 Accessing the Interface

Once launched, the interface will be available at:
- **Local**: http://localhost:7860
- **Public**: A public URL will be provided in the terminal output

## 📖 How to Use

### Single Image Detection
1. Go to the "Single Image Detection" tab
2. Upload an MRI image using the file upload area
3. Click "Detect Tumor" button
4. View the results with bounding boxes and labels
5. Right-click on the result image to download

### Batch Image Processing
1. Go to the "Batch Image Detection" tab
2. Upload multiple MRI images (select multiple files)
3. Click "Process All Images" button
4. View all results in the gallery
5. Download the ZIP file containing all processed images

## 🎯 Detection Classes

The system can detect four types of brain conditions:

| Class | Description |
|-------|-------------|
| **Glioma** | A type of tumor that occurs in the brain and spinal cord |
| **Meningioma** | A tumor that forms on membranes covering the brain and spinal cord |
| **No Tumor** | Normal brain tissue without any tumor |
| **Pituitary** | A tumor in the pituitary gland |

## 🔧 Technical Details

- **Model**: YOLOv10 with CBAM (Convolutional Block Attention Module)
- **Input Format**: JPEG, PNG, BMP images
- **Confidence Threshold**: 50%
- **Processing**: Real-time inference with GPU acceleration
- **Output**: Images with bounding boxes and confidence scores

## 📁 Project Structure

```
fypcode/
├── FYPUI.py              # Main UI application
├── run_ui.py             # Launcher script
├── fyp.py                # Training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── Trained_model/       # Trained model files
│   └── YOLOv10CM_FYPtrained.pt
├── dataset/             # Training dataset
│   ├── train/
│   └── val/
└── yolov10n_CBAM.yaml   # Model configuration
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**:
   - Ensure the model file exists in `Trained_model/YOLOv10CM_FYPtrained.pt`
   - Check file permissions

2. **CUDA/GPU issues**:
   - Install CUDA-compatible PyTorch version
   - Check GPU drivers are up to date

3. **Memory issues**:
   - Reduce batch size in batch processing
   - Process fewer images at once

4. **Port already in use**:
   - Change the port number in the launch parameters
   - Kill existing processes using the port

### Error Messages

- **"Missing required packages"**: Run `pip install -r requirements.txt`
- **"Model file not found"**: Verify model file location
- **"CUDA out of memory"**: Reduce batch size or use CPU processing

## 🔒 Security Notes

- The interface is designed for local use
- When using the public link, be aware that uploaded images are processed on the server
- Consider implementing authentication for production use

## 📊 Performance

- **Single Image**: ~1-3 seconds (GPU), ~5-10 seconds (CPU)
- **Batch Processing**: Varies based on number of images and hardware
- **Memory Usage**: ~2-4 GB RAM recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant medical data privacy regulations when using with real patient data.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the terminal
3. Ensure all dependencies are properly installed

## 🔄 Updates

- **v1.0**: Initial release with single and batch processing
- **v1.1**: Added download functionality and improved UI
- **v1.2**: Enhanced error handling and user experience

---

**Note**: This system is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance may be required.



