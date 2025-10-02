#!/bin/bash
# --- Parse command-line arguments ---
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input_dir=*) input_dir="${1#*=}"; shift ;;
    --category_name=*) category_name="${1#*=}"; shift ;;
    *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
  esac
done

# --- Check if required arguments were provided ---
if [ -z "$input_dir" ]; then
  echo "❌ Error: --input_dir must be specified"
  echo "✅ Usage: ./run_pipeline.sh --input_dir=/path/to/images --category_name=category"
  exit 1
fi

if [ -z "$category_name" ]; then
  echo "❌ Error: --category_name must be specified"
  echo "✅ Usage: ./run_pipeline.sh --input_dir=/path/to/images --category_name=category"
  exit 1
fi

# --- Run pipeline ---
echo "✅ Activating virtual environment..."
source myenv/bin/activate

echo "🚀 Running detect_and_segment.py..."
echo "   Input directory: $input_dir"
echo "   Category name: $category_name"
python3 detect_and_segment.py --input_dir="$input_dir" --category_name="$category_name"

echo "🧠 Running classify.py..."
python3 classify.py

echo "🧹 Deactivating virtual environment..."
deactivate

echo "✅ Done."