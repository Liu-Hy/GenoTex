#!/usr/bin/env python3
"""
Script to recompress specific files back to .gz format.
This is needed for Kaggle users because Kaggle automatically unzips .gz files during dataset import.
"""

import os
import gzip
import shutil

def compress_file(file_path):
    """Compress a file to .gz format and remove the original file."""
    try:
        with open(file_path, 'rb') as f_in:
            with gzip.open(f'{file_path}.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Verify the compressed file exists before removing original
        if os.path.exists(f'{file_path}.gz'):
            os.remove(file_path)
            print(f"Compressed and removed original: {file_path} -> {file_path}.gz")
        else:
            print(f"Warning: Compression succeeded but could not verify {file_path}.gz exists")
    except Exception as e:
        print(f"Error compressing {file_path}: {str(e)}")

def find_and_compress_files(base_dir):
    """Find specific files that need to be compressed and compress them."""
    # Counter for compressed files
    geo_files_count = 0
    tcga_files_count = 0
    
    # Process GEO files
    geo_dir = os.path.join(base_dir, 'input', 'GEO')
    if os.path.exists(geo_dir):
        for trait_dir in os.listdir(geo_dir):
            trait_path = os.path.join(geo_dir, trait_dir)
            if os.path.isdir(trait_path):
                for gse_dir in os.listdir(trait_path):
                    gse_path = os.path.join(trait_path, gse_dir)
                    if os.path.isdir(gse_path):
                        # Look for family.soft files
                        for file in os.listdir(gse_path):
                            if file.endswith('_family.soft'):
                                file_path = os.path.join(gse_path, file)
                                compress_file(file_path)
                                geo_files_count += 1
                            # Look for series_matrix.txt files
                            elif file.endswith('_series_matrix.txt'):
                                file_path = os.path.join(gse_path, file)
                                compress_file(file_path)
                                geo_files_count += 1
    
    # Process TCGA files
    tcga_dir = os.path.join(base_dir, 'input', 'TCGA')
    if os.path.exists(tcga_dir):
        for cancer_dir in os.listdir(tcga_dir):
            cancer_path = os.path.join(tcga_dir, cancer_dir)
            if os.path.isdir(cancer_path):
                # Look for PANCAN files
                for file in os.listdir(cancer_path):
                    if file.endswith('_HiSeqV2_PANCAN'):
                        file_path = os.path.join(cancer_path, file)
                        compress_file(file_path)
                        tcga_files_count += 1
    
    return geo_files_count, tcga_files_count

def main():
    # Use the current directory as base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting to compress files...")
    geo_count, tcga_count = find_and_compress_files(base_dir)
    
    print(f"\nCompression completed!")
    print(f"Compressed {geo_count} GEO files and {tcga_count} TCGA files.")
    print("Original uncompressed files have been removed.")
    print("Please verify the compressed versions work correctly.")

if __name__ == "__main__":
    main()
