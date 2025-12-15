# Archive

This folder contains deprecated files kept for reference.

## Contents

- **`FYPUI.py`** - Original Gradio-based web interface
  - Replaced by FastAPI + React architecture
  - Kept for reference and comparison
  - No longer maintained

## Migration Notes

The project has been migrated from Gradio to a modern FastAPI + React stack:

- **Old (Gradio)**: Single Python file with Gradio UI
- **New**: Separate backend (FastAPI) and frontend (React)

See [MIGRATION_COMPLETE.md](../MIGRATION_COMPLETE.md) for details.

## Using Old Gradio UI (Not Recommended)

If you need to run the old interface:

```bash
python archive/FYPUI.py
```

**Note**: This requires Gradio to be installed and may have hardcoded paths.

---

**Recommended**: Use the new FastAPI + React interface instead:
```bash
python run_ui.py  # Backend
cd frontend && npm run dev  # Frontend
```

