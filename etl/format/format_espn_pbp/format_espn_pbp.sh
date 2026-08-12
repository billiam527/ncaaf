#!/usr/bin/env bash

# Simple error tracking with line numbers
set -eE
trap 'echo "ERROR: Line $LINENO failed with exit code $?" >&2' ERR

echo "ESPN PBP Formatting Script with Expected Points Model"

# Clean and create temp directory
rm -rf temp 2>/dev/null || true
rm -f seasons 2>/dev/null || true
mkdir temp

# Parse arguments
args=$#
if [ $args -eq 1 ]; then
    START_YEAR=$1
    END_YEAR=$1
    echo "Formatting single season: $START_YEAR-$((START_YEAR + 1))"
    echo "$START_YEAR" > temp/seasons
elif [ $args -eq 2 ]; then
    START_YEAR=$1
    END_YEAR=$2
    echo "Formatting year range: $START_YEAR-$END_YEAR"
    
    for i in $(seq $START_YEAR $END_YEAR); do
        echo "$i" >> temp/seasons
    done
else
    echo "Usage: $0 <start_year> [end_year]"
    exit 1
fi

LINES=$(wc -l < temp/seasons)
echo "Processing $LINES seasons..."

# Step 1: Find local PBP files
echo "Step 1: Checking for local PBP files..."
local_files_found=0
local_files_used=0

while IFS= read -r YEAR; do
    SECOND_YEAR=$((YEAR + 1))
    
    local_file="../../collect/collect_espn_pbp/temp/play-by-play_${YEAR}-08-01_to_${SECOND_YEAR}-02-01.csv"
    
    if [ -f "$local_file" ]; then
        echo "   Found local file for $YEAR: $(basename $local_file)"
        absolute_path=$(realpath "$local_file")
        ln -sf "$absolute_path" "temp/pbp_${YEAR}.csv" || {
            echo "ERROR: Failed to create symlink for $YEAR at line $LINENO"
            exit 1
        }
        local_files_found=$((local_files_found + 1))
        local_files_used=$((local_files_used + 1))
    fi
done < temp/seasons

echo "   Found $local_files_found local files"

# Step 2: Download missing files from S3
downloaded_files=0

if [ $local_files_found -lt $LINES ]; then
    echo "Step 2: Downloading missing files from S3..."
    
    while IFS= read -r YEAR; do
        SECOND_YEAR=$((YEAR + 1))
        LOCAL_PBP="temp/pbp_${YEAR}.csv"
        
        if [ ! -f "$LOCAL_PBP" ]; then
            PBP_FILE="play-by-play_${YEAR}-08-01_to_${SECOND_YEAR}-02-01.csv"
            echo "   Downloading: $PBP_FILE"
            
            if aws s3 cp "s3://ncaaf-data/espn-pbp-data/pbp/csvs/$PBP_FILE" "$LOCAL_PBP" 2>&1; then
                downloaded_files=$((downloaded_files + 1))
                echo "   Downloaded successfully"
            else
                echo "   WARNING: Failed to download $PBP_FILE - will skip this year"
            fi
        fi
    done < temp/seasons
    
    echo "   Downloaded $downloaded_files additional files from S3"
else
    echo "Step 2: Using all local files - skipping S3 downloads"
fi

# NEW STEP: Generate Expected Points Lookup Table
echo ""
echo "EXPECTED POINTS MODEL GENERATION"
echo "================================="

# Step 3: Check if we need to generate new EP model or use existing
FORCE_REGENERATE=${FORCE_EP_REGENERATE:-false}
EP_LOOKUP_FILE="expected_points_lookup_table.csv"

if [ "$FORCE_REGENERATE" = true ] || [ ! -f "$EP_LOOKUP_FILE" ]; then
    echo "Step 3: Generating Expected Points model..."
    
    # Check if we have the necessary scripts
    if [ ! -f "expected_points_model.py" ]; then
        echo "ERROR: expected_points_model.py not found at line $LINENO"
        exit 1
    fi
    
    # Run the expected points model to generate initial lookup table
    echo "   Running Expected Points model generation..."
    # Read PIPESTATUS[0] rather than the pipeline status: tee exits 0 even when
    # the model generation fails, which would report a broken run as a success.
    python3 expected_points_model.py 2>&1 | tee temp/ep_model.log
    ep_status=${PIPESTATUS[0]}
    if [ "$ep_status" -eq 0 ]; then
        echo "   SUCCESS: Expected Points model generated"
        
        # Check if the lookup table was created
        if [ ! -f "$EP_LOOKUP_FILE" ]; then
            echo "ERROR: Expected Points lookup table not created at line $LINENO"
            exit 1
        fi
    else
        echo "ERROR: Expected Points model generation failed at line $LINENO"
        echo "   Check temp/ep_model.log for details"
        exit 1
    fi
    
    # Step 4: Detect and smooth anomalies
    echo "Step 4: Detecting and smoothing EP anomalies..."
    
    if [ ! -f "ep_anomaly_detector.py" ]; then
        echo "ERROR: ep_anomaly_detector.py not found at line $LINENO"
        exit 1
    fi
    
    if python3 ep_anomaly_detector.py \
        --input_file "$EP_LOOKUP_FILE" \
        --output_file "expected_points_lookup_table_flagged.csv" \
        --apply_smoothing 2>&1 | tee temp/ep_anomaly.log; then
        echo "   SUCCESS: Anomalies detected and smoothed"
    else
        echo "ERROR: Anomaly detection failed at line $LINENO"
        echo "   Check temp/ep_anomaly.log for details"
        exit 1
    fi
    
    # Step 5: Clean the lookup table
    echo "Step 5: Creating clean Expected Points lookup table..."
    
    if [ ! -f "clean_lookup_table.py" ]; then
        echo "ERROR: clean_lookup_table.py not found at line $LINENO"
        exit 1
    fi
    
    # Backup original lookup table
    cp "$EP_LOOKUP_FILE" "${EP_LOOKUP_FILE}.original" 2>/dev/null || true
    
    if python3 clean_lookup_table.py \
        --input_file "expected_points_lookup_table_flagged.csv" \
        --output_file "$EP_LOOKUP_FILE" 2>&1 | tee temp/ep_clean.log; then
        echo "   SUCCESS: Clean Expected Points lookup table created"
        
        # Show statistics
        if [ -f "$EP_LOOKUP_FILE" ]; then
            rows=$(wc -l < "$EP_LOOKUP_FILE")
            echo "   Final lookup table: $((rows - 1)) scenarios"
        fi
    else
        echo "ERROR: Lookup table cleaning failed at line $LINENO"
        echo "   Check temp/ep_clean.log for details"
        exit 1
    fi
    
else
    echo "Step 3: Using existing Expected Points lookup table"
    if [ -f "$EP_LOOKUP_FILE" ]; then
        rows=$(wc -l < "$EP_LOOKUP_FILE")
        echo "   Found $EP_LOOKUP_FILE with $((rows - 1)) scenarios"
    else
        echo "ERROR: Expected Points lookup file not found at line $LINENO"
        exit 1
    fi
fi

echo ""
echo "PBP PROCESSING WITH EXPECTED POINTS"
echo "===================================="

# Step 6: Process PBP files with EPA using the clean lookup table
echo "Step 6: Processing PBP files with Expected Points Added (EPA)..."
processed_files=0

# Count files and working symlinks
available_files=0
for file in temp/pbp_*.csv; do
    if [ -e "$file" ] && [ -r "$file" ]; then
        available_files=$((available_files + 1))
    fi
done

if [ $available_files -eq 0 ]; then
    echo "ERROR: No PBP files available to process at line $LINENO"
    exit 1
fi

echo "   Processing $available_files PBP files with EPA..."

while IFS= read -r YEAR; do
    LOCAL_PBP="temp/pbp_${YEAR}.csv"
    OUTPUT_FILE="temp/pbp_edit_${YEAR}.csv"
    
    if [ ! -f "$LOCAL_PBP" ]; then
        echo "   Skipping $YEAR (no input file)"
        continue
    fi
    
    echo "   Processing $YEAR with EPA..."
    
    # Use the enhanced edit_pbp_file.py script with EP lookup path.
    #
    # --include_garbage_time keeps garbage-time plays in the output with
    # garbage_time_ind set, rather than deleting them here. Formatting should
    # label plays and let summarize decide what to drop: with the rows removed
    # at this stage, summarize's own --include_garbage_time flag had nothing to
    # act on and the two settings could not be compared.
    if python3 edit_pbp_file.py \
        --input_file "$LOCAL_PBP" \
        --output_file "$OUTPUT_FILE" \
        --ep_lookup_path "$EP_LOOKUP_FILE" \
        --include_epa \
        --include_garbage_time 1 \
        --include_win_prob 2>&1; then
        processed_files=$((processed_files + 1))
        echo "   Enhanced PBP data for $YEAR with EPA"
    else
        error_code=$?
        echo "ERROR: Python processing failed for $YEAR at line $LINENO with exit code $error_code"
        echo "ERROR: Command was: python3 edit_pbp_file.py --input_file $LOCAL_PBP --output_file $OUTPUT_FILE --ep_lookup_path $EP_LOOKUP_FILE"
        
        # Check if Python script exists
        if [ ! -f "edit_pbp_file.py" ]; then
            echo "ERROR: edit_pbp_file.py not found in current directory"
            echo "Current directory: $(pwd)"
            echo "Available Python files:"
            ls -la *.py 2>/dev/null || echo "No Python files found"
        fi
        
        exit 1
    fi
done < temp/seasons

echo "   Successfully processed $processed_files/$available_files files"

if [ $processed_files -eq 0 ]; then
    echo "ERROR: No files were successfully processed at line $LINENO"
    exit 1
fi

# Step 7: Combine processed files
echo "Step 7: Combining processed PBP files..."
pushd temp > /dev/null || {
    echo "ERROR: Cannot enter temp directory at line $LINENO"
    exit 1
}

first_file=$(ls pbp_edit_*.csv 2>/dev/null | head -1)
if [ -z "$first_file" ]; then
    echo "ERROR: No processed files found at line $LINENO"
    popd > /dev/null
    exit 1
fi

echo "   Using header from: $first_file"
head -n 1 "$first_file" > pbp_edit.csv || {
    echo "ERROR: Failed to extract header from $first_file at line $LINENO"
    popd > /dev/null
    exit 1
}

file_count=0
total_rows=1

for file in pbp_edit_*.csv; do
    if [ "$file" != "pbp_edit.csv" ]; then
        file_rows=$(tail -n +2 "$file" | wc -l)
        tail -n +2 "$file" >> pbp_edit.csv || {
            echo "ERROR: Failed to append $file at line $LINENO"
            popd > /dev/null
            exit 1
        }
        file_count=$((file_count + 1))
        total_rows=$((total_rows + file_rows))
    fi
done

echo "   Combined $file_count files into pbp_edit.csv"
echo "   Final file: $total_rows total rows"

popd > /dev/null

# Summary
echo ""
echo "================================================"
echo "PBP FORMATTING WITH EPA COMPLETE!"
echo "================================================"
echo "Expected Points Model:"
if [ "$FORCE_REGENERATE" = true ] || [ ! -f "$EP_LOOKUP_FILE.original" ]; then
    echo "  - Generated new EP model"
    echo "  - Detected and smoothed anomalies"
    echo "  - Created clean lookup table"
else
    echo "  - Used existing EP lookup table"
fi

echo ""
echo "PBP Processing:"
echo "  Files processed: $processed_files"
echo "  Zero-copy optimizations: $local_files_used local files used"  
echo "  S3 downloads: $downloaded_files files"
echo "  Enhanced features: Binary stats, Cumulative stats, EPA, Win Probability"

if [ -f "temp/pbp_edit.csv" ]; then
    final_rows=$(wc -l < temp/pbp_edit.csv)
    echo "  Output: temp/pbp_edit.csv ($final_rows rows)"
    
    # Check if EPA column exists
    if head -n 1 temp/pbp_edit.csv | grep -q "epa"; then
        echo "  EPA: Successfully added to dataset"
    else
        echo "  WARNING: EPA column not found in output"
    fi
    
    echo ""
    echo "SUCCESS: ESPN PBP formatting with EPA completed!"
else
    echo "ERROR: Final output file not found at line $LINENO"
    exit 1
fi

# Optional: Clean up intermediate files
if [ "${CLEANUP_INTERMEDIATE:-false}" = true ]; then
    echo ""
    echo "Cleaning up intermediate files..."
    rm -f expected_points_lookup_table_flagged.csv 2>/dev/null || true
    rm -f expected_points_lookup_table.csv.original 2>/dev/null || true
    echo "   Cleanup complete"
fi