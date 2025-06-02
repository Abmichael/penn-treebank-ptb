#!/usr/bin/env python3
"""Extract zst compressed tar file"""

import tarfile
import zstandard as zstd
import io
from pathlib import Path

def extract_zst_tar(zst_path, extract_to=None):
    """Extract a .tar.zst file"""
    zst_path = Path(zst_path)
    if extract_to is None:
        extract_to = zst_path.parent
    
    print(f"Extracting {zst_path} to {extract_to}")
    
    # Read the zst file
    with open(zst_path, 'rb') as zst_file:
        # Decompress with zstandard
        dctx = zstd.ZstdDecompressor()
        
        # Create a tar file object from the decompressed data
        with dctx.stream_reader(zst_file) as reader:
            # Read all decompressed data into memory (for smaller files)
            decompressed_data = reader.read()
            
            # Create tar file from decompressed data
            tar_buffer = io.BytesIO(decompressed_data)
            
            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                tar.extractall(path=extract_to)
    
    print("Extraction complete!")

if __name__ == "__main__":
    import os
    os.chdir(r"C:\Users\abe\Desktop\402\ML\Proj-2-Penn Treebank (PTB)")
    zst_file = Path("data/ptb/LDC99T42_Penn_Treebank_3.tar.zst")
    print(f"Working directory: {os.getcwd()}")
    print(f"File exists: {zst_file.exists()}")
    if zst_file.exists():
        extract_zst_tar(zst_file)
