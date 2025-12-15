# 🚀 Setup Guide - Brain Tumor Detection System

Complete step-by-step guide to set up and run the application.

## Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
- **Node.js 16+** installed ([Download Node.js](https://nodejs.org/))
- **Git** (optional, for cloning)
- **CUDA-compatible GPU** (optional, but recommended for faster inference)

## Quick Setup (Windows)

### 1. Install Dependencies

**Python packages:**
```bash
pip install -r requirements.txt
```

**Frontend packages:**
```bash
cd frontend
npm install
cd ..
```

### 2. Verify Model File

Ensure your trained model exists at:
```
Trained_model/YOLOv10CM_FYPtrained.pt
```

### 3. Run Tests

```bash
python test_ui.py
```

All tests should pass ✅

### 4. Start the Application

**Option A - Quick Start (Automated):**
```bash
python start_dev.py
```
This will automatically start both backend and frontend servers.

**Option B - Manual Start (Recommended for Development):**

Open **two separate terminals**:

**Terminal 1 (Backend):**
```bash
python run_ui.py
```
Wait for: `Application startup complete.`

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
Wait for: `Local: http://localhost:3000/`

### 5. Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## Quick Setup (Linux/Mac)

### 1. Install Dependencies

**Python packages:**
```bash
pip3 install -r requirements.txt
```

**Frontend packages:**
```bash
cd frontend
npm install
cd ..
```

### 2. Verify Model File

```bash
ls -lh Trained_model/YOLOv10CM_FYPtrained.pt
```

### 3. Run Tests

```bash
python3 test_ui.py
```

### 4. Start the Application

**Terminal 1 (Backend):**
```bash
python3 run_ui.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

### 5. Access the Application

```
http://localhost:3000
```

## Detailed Component Setup

### Backend Setup

The FastAPI backend provides the REST API and model inference.

**Key Files:**
- `backend/app/main.py` - FastAPI application
- `backend/app/inference.py` - Model inference logic
- `backend/app/config.py` - Configuration

**Start Backend:**
```bash
python run_ui.py
```

**Verify Backend:**
```bash
curl http://localhost:8000/api/health
```

**API Documentation:**
```
http://localhost:8000/docs
```

### Frontend Setup

The React frontend provides the user interface.

**Key Files:**
- `frontend/src/App.tsx` - Main application
- `frontend/src/components/` - UI components
- `frontend/src/api.ts` - API client

**Install Dependencies:**
```bash
cd frontend
npm install
```

**Start Development Server:**
```bash
npm run dev
```

**Build for Production:**
```bash
npm run build
```

## Configuration

### Backend Configuration

Edit `backend/app/config.py`:

```python
# Model path (relative to project root)
MODEL_PATH = PROJECT_ROOT / "Trained_model" / "YOLOv10CM_FYPtrained.pt"

# Class names
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Default confidence threshold
DEFAULT_CONFIDENCE = 0.5
```

### Frontend Configuration

Edit `frontend/vite.config.ts` to change the API proxy:

```typescript
export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

### Environment Variables

Create `.env` file in project root (optional):

```bash
# Backend
MODEL_PATH=Trained_model/YOLOv10CM_FYPtrained.pt
DEFAULT_CONFIDENCE=0.5

# Frontend (create frontend/.env)
VITE_API_URL=http://localhost:8000/api
```

## Common Setup Issues

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall Python packages
pip install -r requirements.txt --force-reinstall

# Verify installation
pip list | grep ultralytics
pip list | grep fastapi
```

### Issue: "Model file not found"

**Solution:**
1. Check model exists: `ls -lh Trained_model/`
2. Verify path in `backend/app/config.py`
3. Ensure model filename matches exactly

### Issue: "Port already in use"

**Backend (port 8000):**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Frontend (port 3000):**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:3000 | xargs kill -9
```

### Issue: Frontend can't connect to backend

**Check:**
1. Backend is running on port 8000
2. No CORS errors in browser console
3. Proxy configuration in `vite.config.ts`

**Solution:**
```bash
# Test backend directly
curl http://localhost:8000/api/health

# Check browser console for errors
# F12 -> Console tab
```

### Issue: CUDA/GPU not detected

**Check CUDA:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Solution:**
- Install CUDA toolkit matching your PyTorch version
- Update GPU drivers
- Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Issue: npm install fails

**Solution:**
```bash
cd frontend

# Clear cache
npm cache clean --force

# Remove node_modules
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

## Development Workflow

### Running in Development Mode

1. **Start Backend with Auto-reload:**
   ```bash
   python run_ui.py
   ```
   Changes to Python files will automatically reload the server.

2. **Start Frontend with Hot Module Replacement:**
   ```bash
   cd frontend
   npm run dev
   ```
   Changes to TypeScript/CSS files will instantly update in browser.

### Making Changes

**Backend Changes:**
1. Edit files in `backend/app/`
2. Server auto-reloads
3. Test at http://localhost:8000/docs

**Frontend Changes:**
1. Edit files in `frontend/src/`
2. Browser auto-updates
3. Check browser console for errors

### Testing

**Backend:**
```bash
python test_ui.py
```

**Frontend:**
```bash
cd frontend
npm run build  # Test if it builds successfully
```

**API Testing:**
```bash
# Health check
curl http://localhost:8000/api/health

# Single image detection
curl -X POST http://localhost:8000/api/detect-json \
  -F "file=@test_image.jpg" \
  -F "confidence=0.5"
```

## Production Deployment

### Backend Production

1. **Install production server:**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn:**
   ```bash
   gunicorn backend.app.main:app \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000
   ```

### Frontend Production

1. **Build optimized bundle:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve with a static server:**
   ```bash
   npm install -g serve
   serve -s dist -l 3000
   ```

### Using Docker (Optional)

Create `Dockerfile` for backend:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t brain-tumor-detection .
docker run -p 8000:8000 brain-tumor-detection
```

## CLI Batch Processing

For automated batch processing without the web interface:

```bash
python utils/batch_processor.py input_folder output_folder --confidence 0.5
```

**Example:**
```bash
# Process all images in test_images/
python utils/batch_processor.py test_images/ results/ --confidence 0.6

# Results will be saved to results/ with a JSON summary
```

## Getting Help

If you encounter issues:

1. Check this guide first
2. Run `python test_ui.py` to diagnose
3. Check logs in terminal/console
4. Verify all dependencies: `pip list` and `npm list`
5. Check API docs: http://localhost:8000/docs

## Next Steps

After successful setup:

1. ✅ Test with sample MRI images
2. ✅ Explore API documentation
3. ✅ Try batch processing
4. ✅ Customize confidence thresholds
5. ✅ Review detection results

---

**Setup Complete! 🎉**

Your Brain Tumor Detection System is ready to use!

