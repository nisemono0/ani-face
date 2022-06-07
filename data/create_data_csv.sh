#!/bin/bash

make_csv() {
    images_dir="$1"
    labels_dir="$2"
    out_csv="$3"
    
    images="$(find "$images_dir" -type f -printf '%f\n')"
    labels="$(find "$labels_dir" -type f -printf '%f\n')"

    echo "img,lbl" > "$out_csv"
    paste -d ',' <(printf '%s' "$images") <(printf '%s' "$labels") >> "$out_csv"

}

help_msg="$0 <images_dir> <labels_dir> <out_csv_file>"

if [ "$#" -ne 3 ]; then
    echo "$help_msg"
else
    make_csv "$1" "$2" "$3"
fi
