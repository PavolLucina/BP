import os
import csv

def write_index_map(root_dir):
    """
    Scan root_dir for image files and write index_map.csv with:
    - index: 0-based position in sorted order
    - path: full path to image file
    - label: class name (parent directory name)
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG'}
    paths = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1] in exts:
                paths.append(os.path.join(dirpath, fname))
    
    paths.sort()  # deterministic, lexicographic order
    
    csv_path = os.path.join(root_dir, 'index_map.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'path', 'label'])
        for idx, p in enumerate(paths):
            label = os.path.basename(os.path.dirname(p))
            # Store relative path (relative to root_dir)
            rel_path = os.path.relpath(p, root_dir)
            writer.writerow([idx, rel_path, label])
    
    print(f"wrote {len(paths)} entries to {csv_path}")


if __name__ == '__main__':
    base_path = r"C:\BP\pythonProject1\data_rysy"
    
    # Handle all three dataset versions
    versions = ['', '_detected', '_detected_manual']
    
    for version in versions:
        if version == '_detected_manual':
            # manual variant has extra nesting
            root_dir = os.path.join(base_path, "rys_trening_data_Beno_detected_manual", "rys_trening_data_Beno_detected_manual")
        else:
            dir_suffix = f"rys_trening_data_Beno{version}"
            root_dir = os.path.join(base_path, dir_suffix)
        
        if os.path.isdir(root_dir):
            print(f"Processing version '{version}'...")
            write_index_map(root_dir)
        else:
            print(f"Directory not found: {root_dir}")