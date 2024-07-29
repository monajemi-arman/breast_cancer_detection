#!/usr/bin/env python

import patch
import os

# Input directory
patches_dir = 'patches'

all_files = os.listdir(patches_dir)
patch_files = list(filter(lambda x: x.endswith('.diff'), all_files))

failed = 0
for patch_file in patch_files:
    patch_path = os.path.join(patches_dir, patch_file)
    if patch.fromfile(patch_path).apply():
        print(u'[\N{check mark}] patched successfully: ' + patch_path)
    else:
        failed += 1
        print('[X] patched failed: ' + patch_path)

if failed > 0:
    print("[!] Some File(s) have failed to be patched.")
