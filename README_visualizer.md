# Model Comparison Visualizer

A Gradio application for comparing image segmentation model outputs across different models, with a focus on ADE20K dataset predictions.

## Features

- **Class-based Image Selection**: Filter and find images containing specific classes
- **Multi-Model Comparison**: View predictions from different models side by side
- **Detailed Metrics**: Compare PQ (Panoptic Quality), SQ (Segmentation Quality), and RQ (Recognition Quality) metrics
- **Per-Class Analysis**: View per-class performance metrics for each model
- **Model Difference Visualization**: Compare the differences between models with color-coded metrics
- **Interactive Zooming**: Easily inspect and zoom in on images for detailed analysis

## Getting Started

### Prerequisites

- Python 3.x
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - matplotlib
  - numpy
  - pillow
  - gradio
  - seaborn

### Running the App

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python gradio_app.py
   ```
4. Open your browser at the URL displayed in the terminal (typically `http://127.0.0.1:7860`)

## Usage

1. **Select Classes**: Choose one or more classes from the dropdown menu
2. **Select Models**: Select models you want to compare
3. **Find Images**: Click the "Find Images & Visualize" button to retrieve images containing the selected classes
4. **Navigate Images**: Use the Previous/Next buttons to browse through found images
5. **Jump to Specific Images**: Enter an image index in the text field and click "Go"
6. **View Metrics**: Check the Metrics tab for detailed comparison metrics
7. **Zoom Images**: Use the Zoom Individual Images tab to examine images in detail
   - Hover over images to enlarge slightly
   - Click and hold to zoom in more
   - Click gallery images to view in full screen mode

## Data Structure

The application expects the following data structure:
- Images in `/home/arda/thesis/eomt/output/` with naming convention:
  - Input images: `val_{idx}_input.png`
  - Target segmentations: `val_{idx}_target.png`
  - Model predictions: `val_{idx}_pred_{model_name}.png`
- Class names in `/home/arda/thesis/eomt/output/class_names.txt`
- Metrics and instance counts in `/home/arda/thesis/eomt/output/instance_counts.json`

## Customization

You can customize the app by modifying the file paths in the code to match your directory structure, or by adding additional visualizations as needed. 