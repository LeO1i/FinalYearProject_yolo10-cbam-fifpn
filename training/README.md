# Training Module

This folder contains scripts and configurations for training the YOLOv10 model with CBAM attention mechanism.

## Files

- **`fyp.py`** - Main training script
  - Loads training and validation data
  - Trains YOLOv10 with CBAM modifications
  - Saves trained model to `Trained_model/`

- **`exmodule.py`** - Custom neural network modules
  - `CBAM` - Convolutional Block Attention Module
  - `BiFPN` - Bidirectional Feature Pyramid Network
  - Used in model architecture

- **`yolov10n_CBAM.yaml`** - Model configuration
  - Defines YOLOv10 architecture with CBAM
  - Specifies layer configurations

## Usage

### Training a New Model

1. **Prepare your dataset** in the `dataset/` folder:
   ```
   dataset/
   ├── train/
   │   ├── Glioma/
   │   ├── Meningioma/
   │   ├── No Tumor/
   │   └── Pituitary/
   └── val/
       └── (same structure)
   ```

2. **Run the training script**:
   ```bash
   python training/fyp.py
   ```

3. **Monitor training**:
   - Training progress will be displayed in the terminal
   - Model checkpoints saved to `Trained_model/`

### Configuration

Edit `fyp.py` to adjust training parameters:

```python
result = model.train(
    data=r'D:\fyp\fypcode\dataset.yaml',
    epochs=10,        # Number of epochs
    imgsz=640,        # Image size
    lr0=0.001,        # Initial learning rate
    batch=16,         # Batch size
    # ... more parameters
)
```

### Custom Modules

The `exmodule.py` contains:

#### CBAM (Convolutional Block Attention Module)
- Channel attention via avg/max pooling
- Spatial attention via channel-wise max/avg
- Improves feature representation

#### BiFPN (Bidirectional Feature Pyramid Network)
- Fast normalized fusion
- Weighted feature aggregation
- Better multi-scale feature extraction

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- CUDA-compatible GPU (recommended)

## Notes

- Training can take several hours depending on:
  - Dataset size
  - Number of epochs
  - GPU/CPU performance
  
- Monitor GPU memory usage
- Adjust batch size if out of memory errors occur

## Model Architecture

The model uses YOLOv10 as the base with:
- CBAM attention modules for better feature focus
- BiFPN for multi-scale feature fusion
- Custom head for brain tumor classification

## Classes

The model detects 4 classes:
1. **Glioma** - Brain/spinal cord tumors
2. **Meningioma** - Membrane tumors
3. **No Tumor** - Normal tissue
4. **Pituitary** - Pituitary gland tumors

---

For more information, see the main [README.md](../README.md)

