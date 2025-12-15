# 🧠 Brain Tumor Detection System

A modern web application for detecting brain tumors in MRI images using YOLOv10 with CBAM (Convolutional Block Attention Module). Built with FastAPI backend and React frontend.

## 🚀 Features

- **Single Image Detection**: Upload and process individual MRI images
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Results**: View detection results with bounding boxes and confidence scores
- **Download Functionality**: Download processed images individually or as a ZIP file
- **Modern Web UI**: Responsive React interface with clean, intuitive design
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **CLI Batch Processor**: Command-line tool for batch processing
- **Multiple Tumor Types**: Detects Glioma, Meningioma, No Tumor, and Pituitary tumors

## 📋 Requirements

- Python 3.8 or higher
- Node.js 16+ and npm (for frontend development)
- CUDA-compatible GPU (recommended for faster processing)
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

### Backend Setup

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file exists**:
   Ensure the trained model file is located at:
   ```
   Trained_model/YOLOv10CM_FYPtrained.pt
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

## 🚀 Quick Start

### Development Mode (Recommended)

Run both backend and frontend servers in separate terminals:

**Terminal 1 - Backend:**
```bash
python run_ui.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open your browser to: **http://localhost:3000**

### Production Mode

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Start the backend**:
   ```bash
   python run_ui.py
   ```

3. **Access the application**:
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🌐 API Endpoints

The FastAPI backend provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and model status |
| `/api/detect` | POST | Single image detection (returns image) |
| `/api/detect-json` | POST | Single image detection (returns JSON) |
| `/api/batch` | POST | Batch processing (returns ZIP) |
| `/api/batch-json` | POST | Batch processing (returns JSON) |
| `/docs` | GET | Interactive API documentation |

### API Usage Examples

**Single Image Detection:**
```bash
curl -X POST "http://localhost:8000/api/detect-json" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/api/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence=0.5" \
  -o results.zip
```

## 📖 How to Use

### Web Interface

#### Single Image Detection
1. Go to the "Single Image Detection" tab
2. Upload an MRI image using the file upload button
3. Adjust confidence threshold if needed (default: 50%)
4. Click "Detect Tumor" button
5. View the results with bounding boxes and detection details
6. Download the annotated image using the download link

#### Batch Image Processing
1. Go to the "Batch Image Detection" tab
2. Upload multiple MRI images (select multiple files)
3. Adjust confidence threshold if needed
4. Click "Process All Images" button
5. View detection summary and individual results
6. Download the ZIP file containing all processed images and summary JSON

### Command-Line Batch Processor

For processing multiple images from the command line:

```bash
python utils/batch_processor.py <input_dir> <output_dir> [--confidence 0.5]
```

**Arguments:**
- `input_dir`: Directory containing input images
- `output_dir`: Directory to save processed images
- `--confidence` or `-c`: Confidence threshold (default: 0.5)
- `--no-json`: Skip saving JSON results file

**Example:**
```bash
python utils/batch_processor.py ./input_images ./output_results --confidence 0.6
```

## 🎯 Detection Classes

The system can detect four types of brain conditions:

| Class | Description | Color Code |
|-------|-------------|------------|
| **Glioma** | A type of tumor that occurs in the brain and spinal cord | Yellow |
| **Meningioma** | A tumor that forms on membranes covering the brain and spinal cord | Gray |
| **No Tumor** | Normal brain tissue without any tumor | Green |
| **Pituitary** | A tumor in the pituitary gland | Purple |

## 🔧 Technical Details

### Model
- **Architecture**: YOLOv10 with CBAM (Convolutional Block Attention Module)
- **Input Format**: JPEG, PNG, BMP images
- **Input Size**: 640x640 (automatically resized)
- **Confidence Threshold**: 50% (adjustable)
- **Processing**: Real-time inference with GPU acceleration

### Backend
- **Framework**: FastAPI
- **Server**: Uvicorn with auto-reload in development
- **Image Processing**: OpenCV, PIL
- **Model Inference**: Ultralytics YOLO

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: CSS3 with responsive design
- **API Client**: Native Fetch API

## 📁 Project Structure

```
YOLOV10-cbam-fifpn/
├── backend/                    # Backend application
│   └── app/
│       ├── __init__.py
│       ├── main.py            # FastAPI application
│       ├── config.py          # Configuration and paths
│       ├── inference.py       # Shared inference module
│       └── schemas.py         # Pydantic models
├── frontend/                  # Frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.tsx           # Main app component
│   │   ├── api.ts            # API client
│   │   └── main.tsx          # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── training/                 # Training scripts and configs
│   ├── fyp.py               # Training script
│   ├── exmodule.py          # Custom modules (CBAM, BiFPN)
│   └── yolov10n_CBAM.yaml   # Model configuration
├── Trained_model/            # Trained model files
│   └── YOLOv10CM_FYPtrained.pt
├── dataset/                  # Training dataset
│   ├── train/
│   └── val/
├── archive/                  # Deprecated files
│   └── FYPUI.py             # Old Gradio UI
├── batch_processor.py        # CLI batch processor
├── run_ui.py                 # Application launcher
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found:**
- Ensure the model file exists at `Trained_model/YOLOv10CM_FYPtrained.pt`
- Check file permissions

**2. Backend won't start:**
- Verify all Python dependencies are installed: `pip install -r requirements.txt`
- Check if port 8000 is already in use
- Verify Python version is 3.8 or higher

**3. Frontend won't start:**
- Ensure Node.js 16+ is installed
- Run `npm install` in the frontend directory
- Check if port 3000 is already in use
- Clear npm cache: `npm cache clean --force`

**4. CUDA/GPU issues:**
- Install CUDA-compatible PyTorch version
- Check GPU drivers are up to date
- Model will fall back to CPU if CUDA is unavailable

**5. CORS errors:**
- Ensure backend is running on port 8000
- Check Vite proxy configuration in `frontend/vite.config.ts`

**6. Import errors:**
- Ensure you're running from the project root directory
- Check Python path includes the backend module

### Error Messages

- **"Missing required packages"**: Run `pip install -r requirements.txt`
- **"Model file not found"**: Verify model file location in `Trained_model/`
- **"CUDA out of memory"**: Reduce batch size or use CPU processing
- **"Connection refused"**: Ensure backend server is running
- **"Module not found"**: Check Python path and imports

## 🔒 Security Notes

- The backend allows CORS from all origins by default (development mode)
- For production, configure specific allowed origins in `backend/app/main.py`
- Implement authentication for production deployments
- Validate and sanitize all file uploads
- Consider rate limiting for API endpoints

## 📊 Performance

- **Single Image (GPU)**: ~1-3 seconds
- **Single Image (CPU)**: ~5-10 seconds
- **Batch Processing**: Scales with number of images and hardware
- **Memory Usage**: ~2-4 GB RAM recommended
- **Disk Space**: ~500 MB for model and dependencies

## 🧪 Testing

### Backend Tests
```bash
python test_ui.py
```

### API Health Check
```bash
curl http://localhost:8000/api/health
```

### Frontend Development
```bash
cd frontend
npm run dev
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test thoroughly (backend and frontend)
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant medical data privacy regulations when using with real patient data.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review error messages in terminal/console
3. Check browser developer console for frontend issues
4. Verify all dependencies are properly installed
5. Consult API documentation at http://localhost:8000/docs

## 🔄 Version History

- **v2.0**: Migrated to FastAPI + React architecture
  - Modern web application with separated backend/frontend
  - RESTful API with OpenAPI documentation
  - Improved performance and scalability
  - Better error handling and user experience
  - Maintained CLI batch processor for automation
  
- **v1.2**: Enhanced error handling and user experience (Gradio)
- **v1.1**: Added download functionality and improved UI (Gradio)
- **v1.0**: Initial release with single and batch processing (Gradio)

## 🎓 Model Training

To train the model with your own dataset:

1. Prepare your dataset in YOLO format in the `dataset/` directory
2. Run training:
   ```bash
   python training/fyp.py
   ```

The model configuration is defined in `training/yolov10n_CBAM.yaml` with custom attention modules in `training/exmodule.py`.

**Note:** The training script now uses repo-relative paths, so no path configuration is needed!

See [training/README.md](training/README.md) for detailed training instructions.

---

**Note**: This system is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance may be required.

**Built with ❤️ using YOLOv10, FastAPI, and React**
