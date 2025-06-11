import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse


class DepthAnalysisUtils:
    def __init__(self):
        self.depth_mapping = {'0': 0, '05': 1, '1': 2, '15': 3, '2': 4}
        self.depth_labels = ['0mm', '0.5mm', '1.0mm', '1.5mm', '2.0mm']
        self.object_types = ['metal', 'black', 'ring', 'triangle']
        self.surface_types = ['curved']

    def check_directory_structure(self, base_path='depth'):
        """Check and validate the directory structure"""
        print("Checking directory structure...")
        print("=" * 50)

        if not os.path.exists(base_path):
            print(f"❌ Base directory '{base_path}' not found!")
            return False

        structure_valid = True
        total_images = 0

        for surface_type in self.surface_types:
            surface_path = os.path.join(base_path, surface_type)

            if not os.path.exists(surface_path):
                print(f"❌ Surface directory '{surface_path}' not found!")
                structure_valid = False
                continue

            print(f"\n📁 {surface_type.upper()} SURFACE:")
            surface_total = 0

            for object_type in self.object_types:
                object_path = os.path.join(surface_path, object_type)

                if not os.path.exists(object_path):
                    print(f"  ❌ Object directory '{object_type}' not found!")
                    structure_valid = False
                    continue

                print(f"  📁 {object_type}:")
                object_total = 0

                for depth_folder in ['0', '05', '1', '15', '2']:  # Fixed depth folders
                    depth_path = os.path.join(object_path, depth_folder)

                    if os.path.exists(depth_path):
                        # Count image files
                        image_files = [f for f in os.listdir(depth_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                        count = len(image_files)
                        depth_label = self.depth_labels[self.depth_mapping[depth_folder]]

                        status = "✅" if count > 0 else "⚠️ "
                        print(f"    {status} {depth_label:>6}: {count:>3} images")

                        object_total += count

                        if count == 0:
                            print(f"        Warning: No images found in {depth_path}")
                    else:
                        depth_label = self.depth_labels[self.depth_mapping[depth_folder]]
                        print(f"    ❌ {depth_label:>6}: Directory not found")
                        structure_valid = False

                print(f"    📊 Total {object_type}: {object_total}")
                surface_total += object_total

            print(f"  📊 Total {surface_type} images: {surface_total}")
            total_images += surface_total

        print(f"\n📊 TOTAL IMAGES: {total_images}")

        if structure_valid and total_images > 0:
            print("✅ Directory structure is valid!")
        else:
            print("❌ Directory structure has issues!")

        return structure_valid and total_images > 0

    def create_sample_structure(self, base_path='depth_sample'):
        """Create a sample directory structure for demonstration"""
        print(f"Creating sample directory structure in '{base_path}'...")

        # Create directories
        for surface_type in self.surface_types:
            for object_type in self.object_types:
                for depth_folder in ['0', '05', '1', '15', '2']:
                    dir_path = os.path.join(base_path, surface_type, object_type, depth_folder)
                    os.makedirs(dir_path, exist_ok=True)

        print("✅ Sample directory structure created!")
        print("You can now copy your images into the appropriate folders.")

        # Print structure
        self.print_directory_tree(base_path)

    def print_directory_tree(self, base_path):
        """Print the directory structure as a tree"""
        print(f"\nDirectory structure for '{base_path}':")
        print("📁 " + base_path + "/")
        for surface_idx, surface in enumerate(self.surface_types):
            surface_connector = "└──" if surface_idx == len(self.surface_types) - 1 else "├──"
            print(f"{surface_connector} 📁 {surface}/")

            for obj_idx, obj_type in enumerate(self.object_types):
                obj_connector = "└──" if obj_idx == len(self.object_types) - 1 else "├──"
                obj_prefix = "    " if surface_idx == len(self.surface_types) - 1 else "│   "
                print(f"{obj_prefix}{obj_connector} 📁 {obj_type}/")

                for depth_idx, depth in enumerate(['0', '05', '1', '15', '2']):
                    depth_connector = "└──" if depth_idx == 4 else "├──"
                    depth_prefix = obj_prefix + ("    " if obj_idx == len(self.object_types) - 1 else "│   ")
                    depth_label = self.depth_labels[self.depth_mapping[depth]]
                    print(f"{depth_prefix}{depth_connector} 📁 {depth}/ ({depth_label})")

    def analyze_image_properties(self, base_path='depth', max_samples=5):
        """Analyze properties of images in the dataset"""
        print("Analyzing image properties...")
        print("=" * 50)

        image_info = {
            'shapes': [],
            'dtypes': [],
            'file_sizes': [],
            'sample_paths': []
        }

        for surface_type in self.surface_types:
            surface_path = os.path.join(base_path, surface_type)
            if not os.path.exists(surface_path):
                continue

            for object_type in self.object_types:
                object_path = os.path.join(surface_path, object_type)
                if not os.path.exists(object_path):
                    continue

                for depth_folder in ['0', '05', '1', '15', '2']:
                    depth_path = os.path.join(object_path, depth_folder)
                    if not os.path.exists(depth_path):
                        continue

                    image_files = [f for f in os.listdir(depth_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                    # Sample a few images from each folder
                    sample_files = image_files[:min(max_samples, len(image_files))]

                    for img_file in sample_files:
                        img_path = os.path.join(depth_path, img_file)

                        try:
                            # Load with OpenCV
                            img = cv2.imread(img_path)
                            if img is not None:
                                image_info['shapes'].append(img.shape)
                                image_info['dtypes'].append(str(img.dtype))
                                image_info['file_sizes'].append(os.path.getsize(img_path))
                                image_info['sample_paths'].append(img_path)
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")

        if len(image_info['shapes']) == 0:
            print("❌ No valid images found!")
            return

        # Analyze shapes
        shapes = np.array(image_info['shapes'])
        unique_shapes = np.unique(shapes, axis=0)

        print(f"📊 Analyzed {len(image_info['shapes'])} sample images")
        print(f"\n🖼️  IMAGE SHAPES:")
        for shape in unique_shapes:
            count = np.sum(np.all(shapes == shape, axis=1))
            print(f"  {shape[1]}x{shape[0]}x{shape[2]}: {count} images")

        # Analyze file sizes
        file_sizes = np.array(image_info['file_sizes'])
        print(f"\n💾 FILE SIZES:")
        print(f"  Average: {np.mean(file_sizes) / 1024:.1f} KB")
        print(f"  Min: {np.min(file_sizes) / 1024:.1f} KB")
        print(f"  Max: {np.max(file_sizes) / 1024:.1f} KB")

        # Data types
        unique_dtypes = list(set(image_info['dtypes']))
        print(f"\n🔢 DATA TYPES: {', '.join(unique_dtypes)}")

        return image_info

    def visualize_sample_images(self, base_path='depth', samples_per_class=2):
        """Visualize sample images from each class"""
        print("Creating sample visualization...")

        # Create a grid: 2 surfaces x 5 depths x 4 objects = 40 subplots
        fig, axes = plt.subplots(4, 10, figsize=(25, 12))  # 4 objects x (2 surfaces * 5 depths)
        fig.suptitle('Sample Images from Each Category (Objects x Surface x Depth)', fontsize=16)

        for obj_idx, object_type in enumerate(self.object_types):
            for surf_idx, surface_type in enumerate(self.surface_types):
                for depth_idx, depth_folder in enumerate(['0', '05', '1', '15', '2']):

                    col_idx = surf_idx * 5 + depth_idx  # Column index
                    row_idx = obj_idx  # Row index

                    ax = axes[row_idx, col_idx]

                    depth_path = os.path.join(base_path, surface_type, object_type, depth_folder)

                    if os.path.exists(depth_path):
                        image_files = [f for f in os.listdir(depth_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                        if len(image_files) > 0:
                            # Load first available image
                            img_path = os.path.join(depth_path, image_files[0])
                            img = cv2.imread(img_path)

                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                ax.imshow(img_rgb)

                            depth_label = self.depth_labels[self.depth_mapping[depth_folder]]
                            title = f'{object_type}\n{surface_type}\n{depth_label}'
                            ax.set_title(title, fontsize=8)
                        else:
                            ax.text(0.5, 0.5, 'No Image', ha='center', va='center')
                            ax.set_title(f'{object_type}\n{surface_type}\n{self.depth_labels[depth_idx]}', fontsize=8)
                    else:
                        ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
                        ax.set_title(f'{object_type}\n{surface_type}\n{self.depth_labels[depth_idx]}', fontsize=8)

                    ax.axis('off')

        plt.tight_layout()
        plt.savefig('sample_images_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Sample visualization saved as 'sample_images_grid.png'")

    def prepare_reference_image(self, reference_path, output_path='reference_processed.jpg'):
        """Process the reference image for potential use in analysis"""
        if not os.path.exists(reference_path):
            print(f"❌ Reference image not found: {reference_path}")
            return False

        try:
            # Load reference image
            ref_img = cv2.imread(reference_path)
            if ref_img is None:
                print(f"❌ Could not load reference image: {reference_path}")
                return False

            # Convert to RGB for display
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

            # Display reference image
            plt.figure(figsize=(10, 6))
            plt.imshow(ref_rgb)
            plt.title('Reference Image (Initial State - No Deformation)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('reference_image_display.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Save processed reference
            cv2.imwrite(output_path, ref_img)
            print(f"✅ Reference image processed and saved as '{output_path}'")

            return True

        except Exception as e:
            print(f"❌ Error processing reference image: {e}")
            return False

    def generate_requirements_file(self):
        """Generate requirements.txt file"""
        requirements = [
            "tensorflow>=2.10.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.0.0",
            "numpy>=1.21.0"
        ]

        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))

        print("✅ Requirements file generated: requirements.txt")
        print("Install with: pip install -r requirements.txt")

    def create_run_script(self):
        """Create a simple run script"""
        script_content = """#!/bin/bash
# Depth Analysis Runner Script

echo "Starting Depth Analysis Pipeline..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run training
echo "Starting training..."
python main.py --mode train_depth

# Run testing
echo "Starting testing..."
python main.py --mode test_depth

echo "Analysis complete!"
"""

        with open('run_analysis.sh', 'w') as f:
            f.write(script_content)

        # Make executable on Unix systems
        try:
            import stat
            os.chmod('run_analysis.sh', stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        except:
            pass

        print("✅ Run script created: run_analysis.sh")


def main():
    parser = argparse.ArgumentParser(description='Depth Analysis Utilities')
    parser.add_argument('--check', action='store_true',
                        help='Check directory structure')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample directory structure')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze image properties')
    parser.add_argument('--visualize', action='store_true',
                        help='Create sample visualization')
    parser.add_argument('--setup', action='store_true',
                        help='Set up requirements and run script')
    parser.add_argument('--path', default='depth',
                        help='Base path for depth images (default: depth)')
    parser.add_argument('--reference',
                        help='Path to reference image')

    args = parser.parse_args()

    utils = DepthAnalysisUtils()

    if args.setup:
        utils.generate_requirements_file()
        utils.create_run_script()

    if args.sample:
        utils.create_sample_structure()

    if args.check:
        utils.check_directory_structure(args.path)

    if args.analyze:
        utils.analyze_image_properties(args.path)

    if args.visualize:
        utils.visualize_sample_images(args.path)

    if args.reference:
        utils.prepare_reference_image(args.reference)

    if not any([args.check, args.sample, args.analyze, args.visualize, args.setup, args.reference]):
        print("Depth Analysis Utilities")
        print("=" * 30)
        print("Available commands:")
        print("  --check      : Check directory structure")
        print("  --sample     : Create sample directory structure")
        print("  --analyze    : Analyze image properties")
        print("  --visualize  : Create sample visualization")
        print("  --setup      : Generate requirements and run script")
        print("  --reference  : Process reference image")
        print("  --path PATH  : Specify base path (default: depth)")
        print("\nExample: python depth_utils.py --check --analyze --visualize")


if __name__ == "__main__":
    main()