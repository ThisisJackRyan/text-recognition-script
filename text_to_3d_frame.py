import cv2
import numpy as np
import pytesseract
from PIL import Image
import trimesh
import os

class TextTo3DFrame:
    def __init__(self):
        # Initialize Tesseract OCR with basic configuration
        self.tesseract_config = '--oem 3 --psm 6'
        # Set Tesseract path for macOS
        if os.path.exists('/opt/homebrew/bin/tesseract'):
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
        elif os.path.exists('/usr/local/bin/tesseract'):
            pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        
    def preprocess_image(self, image_path):
        """Preprocess the image for better text recognition."""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Save preprocessed image for debugging
        cv2.imwrite('preprocessed.jpg', dilated)
        
        return dilated

    def extract_text(self, image_path):
        """Extract text from the image using OCR."""
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)
        
        # Perform OCR with better configuration
        text = pytesseract.image_to_string(processed_img, config=self.tesseract_config)
        
        # Get text bounding boxes with confidence scores
        boxes = pytesseract.image_to_data(processed_img, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
        
        return text, boxes

    def create_text_mesh(self, char, position, size, height):
        """Create a 3D mesh for a single character."""
        # Create a simple box for the character
        char_mesh = trimesh.creation.box(extents=[size, size, height])
        char_mesh.apply_translation([position[0], position[1], height/2])
        return char_mesh

    def create_frame_mesh(self, width, height, depth, text_boxes):
        """Create a frame mesh using trimesh."""
        # Create base plate
        base = trimesh.creation.box(extents=[width, height, 2])
        base.apply_translation([0, 0, 0])
        
        # Create the frame sides
        # Bottom
        bottom = trimesh.creation.box(extents=[width, height, 2])
        bottom.apply_translation([0, 0, 0])
        
        # Top
        top = trimesh.creation.box(extents=[width, height, 2])
        top.apply_translation([0, 0, depth-2])
        
        # Left side
        left = trimesh.creation.box(extents=[2, height, depth])
        left.apply_translation([0, 0, depth/2])
        
        # Right side
        right = trimesh.creation.box(extents=[2, height, depth])
        right.apply_translation([width-2, 0, depth/2])
        
        # Front side
        front = trimesh.creation.box(extents=[width-4, 2, depth])
        front.apply_translation([2, 0, depth/2])
        
        # Back side
        back = trimesh.creation.box(extents=[width-4, 2, depth])
        back.apply_translation([2, height-2, depth/2])
        
        # Create inner cutout
        inner = trimesh.creation.box(extents=[width-20, height-20, depth+2])
        inner.apply_translation([10, 10, -1])
        
        # Create text meshes
        text_meshes = []
        if text_boxes:
            for i, (char, conf) in enumerate(zip(text_boxes['text'], text_boxes['conf'])):
                if conf > 60:  # Only use characters with high confidence
                    x = text_boxes['left'][i] * (width / 1000)  # Scale coordinates
                    y = text_boxes['top'][i] * (height / 1000)
                    w = text_boxes['width'][i] * (width / 1000)
                    h = text_boxes['height'][i] * (height / 1000)
                    
                    # Create raised text
                    text_mesh = self.create_text_mesh(char, [x, y], min(w, h), 5)  # 5mm height
                    text_meshes.append(text_mesh)
        
        # Combine all parts
        frame = trimesh.util.concatenate([bottom, top, left, right, front, back])
        
        # Create a scene with all components
        scene = trimesh.Scene([base, frame, inner] + text_meshes)
        
        return scene

    def create_3d_frame(self, extracted_text, boxes, output_path, frame_width=200, frame_height=200, frame_depth=10):
        """Create a 3D model of the frame with raised text."""
        try:
            # Create the frame mesh with text
            scene = self.create_frame_mesh(frame_width, frame_height, frame_depth, boxes)
            
            # Export the scene to STL
            scene.export(output_path)
            
            print(f"3D frame saved to: {output_path}")
            print(f"Extracted text: {extracted_text}")
        except Exception as e:
            print(f"Error creating 3D frame: {str(e)}")
            raise

    def process_image(self, image_path, output_path):
        """Process the image and create a 3D frame."""
        # Extract text and boxes
        extracted_text, boxes = self.extract_text(image_path)
        
        # Create 3D frame
        self.create_3d_frame(extracted_text, boxes, output_path)
        
        return extracted_text

def main():
    # Example usage
    processor = TextTo3DFrame()
    
    # Get input image path from user
    image_path = input("Enter the path to your image: ")
    
    # Set output path to Downloads folder with .stl extension
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.stl'
    output_path = os.path.expanduser(f'~/Downloads/{output_filename}')
    
    try:
        # Process the image
        extracted_text = processor.process_image(image_path, output_path)
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 