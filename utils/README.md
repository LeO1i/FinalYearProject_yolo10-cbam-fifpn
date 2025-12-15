# Utilities

Command-line tools and utility scripts for the Brain Tumor Detection System.

## Contents

### `batch_processor.py`

Command-line batch processor for processing multiple MRI images.

**Usage:**
```bash
python utils/batch_processor.py <input_dir> <output_dir> [options]
```

**Arguments:**
- `input_dir` - Directory containing input images
- `output_dir` - Directory to save processed images
- `--confidence` or `-c` - Confidence threshold (default: 0.5)
- `--no-json` - Skip saving JSON results file

**Example:**
```bash
python utils/batch_processor.py ./test_images ./results --confidence 0.6
```

**Output:**
- Processed images with bounding boxes (in `output_dir`)
- JSON summary file with detection results
- Console output with processing status

## Features

- Batch processing of multiple images
- Progress tracking
- Error handling for invalid images
- JSON export of detection results
- Configurable confidence threshold
- Uses shared inference module (same as API)

## Requirements

All requirements are in the main `requirements.txt` file.

---

For more information, see the main [README.md](../README.md)

