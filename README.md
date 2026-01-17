# Screened Poisson Surface Reconstruction

Implemention of **Screened Poisson Surface Reconstruction** algorithm. It offers two modes:
1. **Manual Python Implementation** (`--manual`): A custom implementation from scratch using a dense grid solver (Splatting -> Divergence -> Laplacian -> Solve -> Marching Cubes).
2. **Open3D Wrapper** (Default): Uses the highly optimized C++ implementation from Open3D.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the `main.py` script. Outputs are automatically saved to the `data/` directory.

### 1. Manual Implementation (Python)
Recommended for understanding the algorithm. Use `depth <= 6` for speed, `depth=7` for higher quality (slower).

```bash
# Reconstruct a synthetic sphere
python main.py --manual --depth 6

# Reconstruct a custom PLY file (e.g., bunny)
python main.py --input path/to/file.ply --manual --depth 6
```

**Options:**
- `--manual`: Use the custom Python solver.
- `--decimate N`: Simplify the output mesh to N triangles (e.g., `--decimate 5000`).
- `--visualize`: Show the result in a 3D window.

### 2. Standard Implementation (Open3D)
Fast and robust for high resolutions.

```bash
python main.py --input path/to/file.ply --depth 9
```

## Project Structure

- `main.py`: CLI entry point.
- `src/`:
  - `pysr/`: **Manual Implementation Core**
    - `formulation.py`: System assembly (Laplacian, Divergence).
    - `solver.py`: Linear system solver and Iso-surface extraction.
  - `preprocess.py`: Point cloud loading and normal estimation.
  - `reconstruction.py`: High-level wrapper for both methods.
  - `postprocess.py`: Mesh cleaning.
