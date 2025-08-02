#!/bin/bash
for f in $(find . -name "*.tex"); do
    echo -e "\n\n--- Building $f"
    sqlplot-tools "$f" || exit 1
done

