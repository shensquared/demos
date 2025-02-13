#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <filename_without_extension>"
  exit 1
fi

# Get the file basename from the first argument
filename="$1"

# Convert PDF to PNG using ImageMagick
convert -density 300 "${filename}.pdf" -trim -quality 90 "${filename}.png"

echo "Conversion complete: ${filename}.png created."