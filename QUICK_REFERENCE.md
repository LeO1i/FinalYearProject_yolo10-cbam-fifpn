# Quick Reference Card

## 🚀 Starting the Application

### Development Mode (Recommended)
```bash
# Terminal 1 - Backend
python run_ui.py

# Terminal 2 - Frontend  
cd frontend && npm run dev

# Open browser: http://localhost:3000
```

### Quick Start (Both servers)
```bash
python start_dev.py
```

## 🔗 URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/health |

## 📁 Key Files & Directories

```
Project Root
├── backend/app/
│   ├── main.py          # FastAPI routes
│   ├── inference.py     # Model inference
│   ├── config.py        # Configuration
│   └── schemas.py       # API models
├── frontend/src/
│   ├── App.tsx          # Main app
│   ├── api.ts           # API client
│   └── components/      # UI components
├── Trained_model/       # Model files
├── run_ui.py           # Backend launcher
└── batch_processor.py  # CLI tool
```

## 🔧 Common Commands

### Backend
```bash
# Start server
python run_ui.py

# Run tests
python test_ui.py

# CLI batch processing
python utils/batch_processor.py input/ output/ --confidence 0.5
```

### Frontend
```bash
# Install dependencies
cd frontend && npm install

# Development server
npm run dev

# Production build
npm run build

# Preview build
npm run preview
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/detect` | POST | Single image (returns image) |
| `/api/detect-json` | POST | Single image (returns JSON) |
| `/api/batch` | POST | Batch processing (returns ZIP) |
| `/api/batch-json` | POST | Batch processing (returns JSON) |

### API Example
```bash
# Single detection
curl -X POST http://localhost:8000/api/detect-json \
  -F "file=@image.jpg" \
  -F "confidence=0.5"

# Batch with ZIP download
curl -X POST http://localhost:8000/api/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -o results.zip
```

## 🎯 Detection Classes

- **Glioma** - Brain/spinal cord tumor
- **Meningioma** - Membrane tumor
- **No Tumor** - Normal tissue
- **Pituitary** - Pituitary gland tumor

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Port in use | `lsof -ti:8000 \| xargs kill -9` (Mac/Linux)<br>`netstat -ano \| findstr :8000` (Windows) |
| Model not found | Check `Trained_model/YOLOv10CM_FYPtrained.pt` exists |
| CORS errors | Ensure backend is running |
| npm errors | `npm cache clean --force && npm install` |

## ⚙️ Configuration

### Change Backend Port
Edit `run_ui.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Change Frontend Port
Edit `frontend/vite.config.ts`:
```typescript
server: { port: 3000 }
```

### Change Confidence Threshold
Edit `backend/app/config.py`:
```python
DEFAULT_CONFIDENCE = 0.5  # 50%
```

## 📊 Model Information

- **Architecture**: YOLOv10 + CBAM
- **Input Size**: 640x640 (auto-resized)
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Format**: PyTorch (.pt)

## 🔑 Keyboard Shortcuts

### Frontend
- `Ctrl/Cmd + Shift + I` - Open DevTools
- `F5` - Refresh page
- `Ctrl/Cmd + R` - Reload without cache

### Backend
- `Ctrl + C` - Stop server
- `Ctrl + Z` - Suspend process

## 📦 Dependencies

### Backend (Python)
- fastapi
- uvicorn
- ultralytics
- opencv-python
- pillow
- numpy

### Frontend (Node.js)
- react
- typescript
- vite
- axios

## 🔍 Debugging

### Check Backend Logs
```bash
# Terminal running python run_ui.py
# Look for errors and warnings
```

### Check Frontend Console
```javascript
// Browser DevTools (F12) -> Console
// Look for API errors
```

### Test API Health
```bash
curl http://localhost:8000/api/health
```

### Test Model Loading
```python
python -c "from backend.app.inference import load_model; load_model()"
```

## 💡 Tips

1. **Always run both backend and frontend** for the full application
2. **Check terminal output** for errors
3. **Use API docs** for testing: http://localhost:8000/docs
4. **Clear browser cache** if UI doesn't update
5. **Restart servers** after config changes

## 📱 Features

✅ Single image detection  
✅ Batch processing  
✅ ZIP download  
✅ Adjustable confidence  
✅ Real-time preview  
✅ Detection summary  
✅ CLI batch tool  
✅ REST API  

## 🎓 Training (Optional)

```bash
# Train new model
python training/fyp.py

# Model config
training/yolov10n_CBAM.yaml

# Custom modules
training/exmodule.py  # CBAM, BiFPN
```

---

**Pro Tip**: Keep this reference open while developing! 🚀

