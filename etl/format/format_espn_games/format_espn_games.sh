#!/usr/bin/env bash
set -e

echo "ESPN Games Formatting Script (Optimized)"

# Clean and create temp directory
rm -rf temp 2>/dev/null || true
mkdir temp

# Parse arguments
args=$#
if [ $args -eq 1 ]; then
    START_YEAR=$1
    END_YEAR=$((START_YEAR + 1))
    echo "Formatting single season: $START_YEAR-$END_YEAR"
elif [ $args -eq 2 ]; then
    START_YEAR=$1
    END_YEAR=$2
    echo "Formatting year range: $START_YEAR-$END_YEAR"
else
    echo "Usage: $0 <start_year> [end_year]"
    exit 1
fi

# ==============================================
# ENHANCED ZERO-COPY OPTIMIZATION
# ==============================================

echo "Step 1: Enhanced local file discovery..."
years=$(seq $START_YEAR $END_YEAR)
local_files_found=0
local_files_used=0

# Load local files map if available (from parent format script)
declare -A local_files
if [ -f "temp/local_files.map" ]; then
    while IFS='=' read -r key value; do
        local_files["$key"]="$value"
    done < temp/local_files.map
fi

# Multiple search paths for local files
search_paths=(
    "../../collect/collect_espn_games/temp"
    "../collect/collect_espn_games/temp"
    "../../collect_espn_games/temp"
    "../collect_espn_games/temp"
)

years=$(seq $START_YEAR $END_YEAR)
for year in $years; do
    SECOND_YEAR=$((year + 1))
    target_file="temp/games_${year}.csv"
    found_file=""
    
    # Check local files map first
    games_key="games:games_${year}-08-01_to_${SECOND_YEAR}-02-01.csv"
    if [ -n "${local_files[$games_key]}" ] && [ -f "${local_files[$games_key]}" ]; then
        found_file="${local_files[$games_key]}"
        echo "   Map link for $year: $(basename $found_file)"
    else
        # Search in known paths
        for path in "${search_paths[@]}"; do
            candidate="$path/games_${year}-08-01_to_${SECOND_YEAR}-02-01.csv"
            if [ -f "$candidate" ]; then
                found_file="$candidate"
                echo "   Found local file for $year: $(basename $found_file)"
                break
            fi
        done
    fi
    
    if [ -n "$found_file" ]; then
        # Create zero-copy symlink with absolute path
        absolute_path=$(realpath "$found_file")
        ln -sf "$absolute_path" "$target_file"
        local_files_found=$((local_files_found + 1))
        local_files_used=$((local_files_used + 1))
    fi
done

echo "   SUCCESS: Found $local_files_found local files (using zero-copy links)"

# ==============================================
# OPTIMIZED S3 BATCH DOWNLOAD
# ==============================================

missing_years=()
for year in $years; do
    if [ ! -f "temp/games_${year}.csv" ]; then
        missing_years+=($year)
    fi
done

downloaded_files=0
if [ ${#missing_years[@]} -gt 0 ]; then
    echo "Step 2: Batch downloading ${#missing_years[@]} missing files from S3..."
    
    # Get S3 file list once. An empty prefix makes grep exit 1, which under
    # `set -e` would kill the script here, so tolerate no matches.
    s3_files=$(aws s3 ls s3://ncaaf-data/espn-games-data/games/csvs/ --recursive | awk '{print $4}' | grep -v '^$' || true)

    # Download sequentially and explicitly. The previous version piped a
    # "src:dest" list through `sed 's/:/ /'` (which split on the colon inside
    # "s3://") into `xargs bash -c` (where the first argument binds to $0, not
    # $1). Both bugs made the copy fail, and the non-zero xargs exit took the
    # whole script down via set -e before the combine step could run - leaving
    # no games.csv and silently falling back to stale data downstream.
    for year in "${missing_years[@]}"; do
        found_match=false
        for s3_file in $s3_files; do
            # Anchor on the year prefix: a loose *"$year"* match makes 2026
            # match games_2025-08-01_to_2026-02-01.csv and duplicate a season.
            if [[ $(basename "$s3_file") == games_${year}-* ]]; then
                src="s3://ncaaf-data/${s3_file}"
                dest="temp/games_${year}.csv"
                if aws s3 cp "$src" "$dest" --quiet; then
                    echo "   SUCCESS: Downloaded: $(basename "$dest")"
                    downloaded_files=$((downloaded_files + 1))
                else
                    echo "   WARNING: Download failed for $year - skipping that year"
                fi
                found_match=true
                break
            fi
        done
        if [ "$found_match" = false ]; then
            echo "   INFO: No S3 file for $year (likely a season that has not started) - skipping"
        fi
    done

    echo "   Downloaded $downloaded_files file(s) from S3"
else
    echo "Step 2: All files available locally - skipping S3 downloads"
fi

# Verify we have files to work with
# Count files and working symlinks
available_files=0
for file in temp/games_*.csv; do
    if [ -e "$file" ] && [ -r "$file" ]; then
        available_files=$((available_files + 1))
    fi
done
if [ $available_files -eq 0 ]; then
    echo "ERROR: No game files found for the specified years"
    exit 1
fi

echo "   Total files available: $available_files"

# ==============================================
# STREAMING FILE COMBINATION
# ==============================================

echo "Step 3: Streaming file combination..."

# Create Python streaming combiner
cat > temp/combine_games.py << 'PYTHON_EOF'
import pandas as pd
import glob
import sys
import os

def combine_games_streaming(input_pattern, output_file, chunk_size=50000):
    """Combine game files using streaming to handle large datasets"""
    
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"ERROR: No files found matching pattern: {input_pattern}")
        return False
    
    print(f"Combining {len(files)} files...")
    
    first_chunk = True
    total_rows = 0
    
    for i, file_path in enumerate(files):
        print(f"   Processing ({i+1}/{len(files)}): {os.path.basename(file_path)}")
        
        try:
            # Read file in chunks to handle large files
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                if chunk.empty:
                    continue
                
                # Write chunk
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                
                chunk.to_csv(
                    output_file,
                    mode=mode,
                    header=header,
                    index=False
                )
                
                first_chunk = False
                total_rows += len(chunk)
                
        except Exception as e:
            print(f"   WARNING: Error processing {file_path}: {e}")
            continue
    
    print(f"   SUCCESS: Combined into {total_rows:,} total rows")
    return total_rows > 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combine_games.py <input_pattern> <output_file>")
        sys.exit(1)
    
    input_pattern = sys.argv[1]
    output_file = sys.argv[2]
    
    if combine_games_streaming(input_pattern, output_file):
        print(f"SUCCESS: Combined files written to {output_file}")
    else:
        print("ERROR: Failed to combine files")
        sys.exit(1)

PYTHON_EOF

# Run streaming combiner
if python3 temp/combine_games.py "temp/games_*.csv" "temp/games_combined.csv"; then
    echo "   SUCCESS: Streaming combination completed"
else
    echo "   ERROR: File combination failed"
    exit 1
fi

# ==============================================
# OPTIMIZED CALENDAR CREATION
# ==============================================

echo "Step 4: Creating calendar data..."

# Extract date range efficiently
echo "   Analyzing date range..."
python3 << 'PYTHON_EOF'
import pandas as pd
import sys

try:
    # Read just the date column to find range
    df = pd.read_csv('temp/games_combined.csv', usecols=['date'], nrows=None)
    
    if df.empty:
        print("ERROR: No data found in games file")
        sys.exit(1)
    
    # Convert to datetime and find range
    df['date'] = pd.to_datetime(df['date'])
    min_year = df['date'].dt.year.min()
    max_year = df['date'].dt.year.max()
    
    print(f"Date range: {min_year} to {max_year}")
    
    # Save range to file for shell script
    with open('temp/date_range.txt', 'w') as f:
        f.write(f"{min_year}\n{max_year}\n")
    
except Exception as e:
    print(f"ERROR: Error analyzing dates: {e}")
    sys.exit(1)

PYTHON_EOF

# Read date range
if [ -f temp/date_range.txt ]; then
    MINYEAR=$(sed -n '1p' temp/date_range.txt)
    MAXYEAR=$(sed -n '2p' temp/date_range.txt)
    echo "   Date range found: $MINYEAR to $MAXYEAR"
else
    echo "ERROR: Could not determine date range"
    exit 1
fi

# Create calendar data
echo "   Generating calendar..."
if python create_calendar.py --start_date "${MINYEAR}-01-01" --end_date "${MAXYEAR}-12-31"; then
    echo "   SUCCESS: Calendar data created"
else
    echo "ERROR: Failed to create calendar data"
    exit 1
fi

# ==============================================
# OPTIMIZED DATA MERGE
# ==============================================

echo "Step 5: Merging with calendar data..."

# Use the combined file for merging
if python merge_dfs.py \
    --file1 temp/games_combined.csv \
    --file2 "temp/schedule_${MINYEAR}_to_${MAXYEAR}.csv" \
    --on 'date' \
    --how 'left' \
    --fillna 'bowl'; then
    echo "   SUCCESS: Merged games with calendar data"
else
    echo "ERROR: Failed to merge data"
    exit 1
fi

# ==============================================
# FINALIZE AND PREPARE FOR UPLOAD
# ==============================================

echo "Step 6: Finalizing results..."
final_file="temp/games.csv"

if [ -f temp/new_games.csv ]; then
    mv temp/new_games.csv "$final_file"
elif [ -f temp/games_combined.csv ]; then
    mv temp/games_combined.csv "$final_file"
else
    echo "ERROR: No final games file created"
    exit 1
fi

final_rows=$(wc -l < "$final_file")

# Check if running from master script (batch upload mode)
if [ "$SKIP_S3_UPLOAD" = true ]; then
    echo ""
    echo "=========================================="
    echo "S3 UPLOAD DEFERRED TO BATCH"
    echo "=========================================="
    echo "Files saved locally for batch upload:"
    
    if [ -f "$final_file" ]; then
        echo "  games.csv ($final_rows rows)"
        echo "  Location: $(pwd)/$final_file"
        
        # Create local backup anyway (fast)
        mkdir -p ../data/games/backups 2>/dev/null || true
        cp "$final_file" ../data/games/backups/games_formatted.csv 2>/dev/null || true
        echo "  Local backup: ../data/games/backups/"
    fi
    
    echo "Batch upload will handle S3 transfer efficiently"
    
else
    # STANDALONE MODE: Do upload when script runs alone
    echo "Step 7: Uploading to S3..."
    if aws s3 cp "$final_file" s3://ncaaf-data/espn-games-data/games/csvs/games.csv; then
        echo "   SUCCESS: Uploaded to S3"
    else
        echo "   WARNING: Upload failed (continuing anyway)"
    fi
fi

# ==============================================
# SUMMARY AND CLEANUP
# ==============================================

echo ""
echo "SUCCESS: GAMES FORMATTING COMPLETE!"
echo "================================"
echo "Final file: $final_rows rows"
echo "Zero-copy optimizations: $local_files_used local files used"
echo "S3 downloads: $downloaded_files files"
echo "Streaming processing: Enabled"
echo "Output: $final_file"

# Cleanup temp files (keep the final output)
rm -f temp/games_*.csv temp/games_combined.csv temp/combine_games.py temp/date_range.txt 2>/dev/null || true

echo "SUCCESS: ESPN games formatting completed successfully!"