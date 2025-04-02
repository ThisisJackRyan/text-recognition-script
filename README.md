# Text to 3D Frame Generator

This Python application converts text from images (both handwritten and typed) into 3D printable frames. The text is extracted using OCR and then converted into a raised 3D model within a picture frame.

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR engine
3. OpenSCAD (for viewing and exporting the 3D models)

### Installing Tesseract OCR

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr
```

#### Windows
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. Clone this repository
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
#### windows 
```bash
python text_to_3d_frame.py
```

#### macOS 
```bash
python3 text_to_3d_frame.py
```

2. When prompted, enter:
   - The path to your input image
   - The desired output path for the SCAD file

3. The script will:
   - Process the image and extract text
   - Create a 3D model with raised text
   - Save the model as a SCAD file

4. Open the generated SCAD file in OpenSCAD to:
   - Preview the 3D model
   - Export to STL format for 3D printing

## Features

- Supports both handwritten and typed text
- Creates a customizable picture frame
- Raises the extracted text for 3D printing
- Preserves text positioning and style
- Adjustable frame dimensions and text height

## Notes

- For best results, use clear, well-lit images
- Text should be clearly visible against the background
- The frame dimensions can be adjusted in the code (default: 200x200x10mm)
