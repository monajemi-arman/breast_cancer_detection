#!/bin/bash

# Input image directory: images_in/
# Labels JSON: labels.json

# Create output directories if they don't exist
mkdir -p images_pred images_gt

# Process all JPEG files in the input directory
for img_path in images_in/*.jpg; do
    # Get just the filename without path
    filename=$(basename "$img_path")
    
    # Send request to API and capture response
    response=$(curl -s -X POST \
        -F "file=@$img_path" \
        -F "gt_file=@labels.json" \
        http://localhost:33517/api/v1/predict)
    
    # Extract and decode ground truth image
    echo "$response" | jq -r '.data.ground_truth_image' | base64 -d > "images_gt/$filename"
    
    # Extract and decode inferred image
    echo "$response" | jq -r '.data.inferred_image' | base64 -d > "images_pred/$filename"
    
    echo "Processed: $filename"
done
