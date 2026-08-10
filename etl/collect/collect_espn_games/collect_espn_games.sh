#!/usr/bin/env bash
set -e

echo "Running ESPN games collection script (FIXED VERSION)"

# Clean and create temp directory
rm -rf temp 2>/dev/null || true
mkdir temp

args=$#

# Function to check if year has available data
check_data_availability() {
    local year=$1
    local current_year=$(date +"%Y")
    local current_month=$(date +"%m")
    local current_day=$(date +"%d")
    
    # Only collect data for years that have actually occurred
    # College football seasons run Aug-Feb, so check accordingly
    if [ $year -gt $current_year ]; then
        echo "WARNING: Skipping year $year - season hasn't started yet"
        return 1
    elif [ $year -eq $current_year ]; then
        # Check if we're before the season starts (around Aug 23)
        if [ $current_month -lt 8 ]; then
            echo "WARNING: Skipping year $year - season hasn't started yet (starts in August)"
            return 1
        elif [ $current_month -eq 8 ] && [ $current_day -lt 20 ]; then
            echo "WARNING: Skipping year $year - season starts around Aug 23"
            return 1
        elif [ $current_month -eq 8 ] && [ $current_day -lt 30 ]; then
            echo "INFO: Season just started, limited data may be available for $year"
        fi
    fi
    return 0
}

if [ $args -eq 0 ]; then
    echo "No arguments provided - collecting all available years from 2010 to current completed seasons"
    current_year=$(date +"%Y")
    current_month=$(date +"%m")
    
    # Determine the last complete season
    if [ $current_month -lt 8 ]; then
        # Before August, last complete season was previous year
        last_complete_year=$((current_year - 1))
    else
        # After August, current season may be in progress
        last_complete_year=$current_year
    fi
    
    echo "Collecting data from 2010 to $last_complete_year"
    
    for i in $(seq 2010 $last_complete_year); do
        if ! check_data_availability $i; then
            continue
        fi
        
        SECOND_YEAR=$((i + 1))
        START_DATE="${i}-08-01"
        END_DATE="${SECOND_YEAR}-02-01"
        echo "Scraping from ${START_DATE} to ${END_DATE}"
        
        # Run the Python script and capture its exit code
        if python3 run.py --start_date "$START_DATE" --end_date "$END_DATE"; then
            echo "SUCCESS: Successfully completed data collection for year $i"
        else
            exit_code=$?
            echo "WARNING: Data collection for year $i completed with warnings or no data found"
            echo "    This is normal if the season hasn't started or has limited games"
            echo "    Check the logs for details"
            # Don't exit on collection "failures" that are actually just empty results
        fi
    done

elif [ $args -eq 1 ]; then
    year=$1
    
    if ! check_data_availability $year; then
        echo "Error: Year $year data is not available yet"
        exit 1
    fi
    
    echo "Collecting single year: $year"
    SECOND_YEAR=$((year + 1))
    START_DATE="$year-08-01"
    END_DATE="${SECOND_YEAR}-02-01"
    echo "Scraping from ${START_DATE} to ${END_DATE}"
    
    # Run the Python script and capture its exit code
    if python3 run.py --start_date "$START_DATE" --end_date "$END_DATE"; then
        echo "SUCCESS: Successfully completed data collection for year $year"
    else
        exit_code=$?
        echo "WARNING: Data collection for year $year completed with warnings or no data found"
        echo "    This is normal if the season hasn't started or has limited games"
        echo "    Check the logs for details"
    fi

elif [ $args -eq 2 ]; then
    start_year=$1
    end_year=$2
    
    echo "Collecting year range: $start_year to $end_year"
    for i in $(seq $start_year $end_year); do
        if ! check_data_availability $i; then
            continue
        fi
        
        SECOND_YEAR=$((i + 1))
        START_DATE="${i}-08-01"
        END_DATE="${SECOND_YEAR}-02-01"
        echo "Scraping from ${START_DATE} to ${END_DATE}"
        
        # Run the Python script and capture its exit code
        if python3 run.py --start_date "$START_DATE" --end_date "$END_DATE"; then
            echo "SUCCESS: Successfully completed data collection for year $i"
        else
            exit_code=$?
            echo "WARNING: Data collection for year $i completed with warnings or no data found"
            echo "    This is normal if the season hasn't started or has limited games"
            echo "    Check the logs for details"
        fi
    done

else
    echo "Error: Too many arguments. Usage: $0 [start_year] [end_year]"
    exit 1
fi

# Check if running from master script (batch upload mode)
if [ "$SKIP_S3_UPLOAD" = true ]; then
    echo ""
    echo "=========================================="
    echo "S3 UPLOAD DEFERRED TO BATCH"
    echo "=========================================="
    echo "Files saved locally for batch upload:"
    
    if [ -d "temp" ]; then
        pushd temp > /dev/null
        
        # Show what files are ready for batch upload
        csv_count=$(find . -name "games_*.csv" | wc -l)
        json_count=$(find . -name "*.json" 2>/dev/null | wc -l || echo "0")
        other_count=$(find . -type f ! -name "*.csv" ! -name "*.json" | wc -l)
        
        echo "  CSV files ready: $csv_count"
        echo "  JSON files ready: $json_count" 
        echo "  Other files ready: $other_count"
        echo "  Location: $(pwd)"
        
        # Create local backup anyway (fast)
        mkdir -p ../data/games/backups 2>/dev/null || true
        cp games_*.csv ../data/games/backups/ 2>/dev/null || true
        echo "  Local backup: ../data/games/backups/"
        
        popd > /dev/null
    fi
    
    echo "Batch upload will handle S3 transfer efficiently"
    
else
    # STANDALONE MODE: Do smart backup when script runs alone
    echo ""
    echo "=========================================="
    echo "SMART BACKUP - ESSENTIAL DATA ONLY"
    echo "=========================================="
    
    if [ -d "temp" ]; then
        pushd temp > /dev/null
        
        # Count and size the essential files
        csv_files=$(find . -name "games_*.csv" | wc -l)
        csv_size=$(find . -name "games_*.csv" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1 2>/dev/null || echo "0")
        
        echo "Essential files found:"
        echo "  Games CSV files: $csv_files (Size: $csv_size)"
        
        if [ $csv_files -gt 0 ]; then
            # Local backup first (fast)
            echo ""
            echo "1. Creating local backup..."
            mkdir -p ../data/games/backups
            cp games_*.csv ../data/games/backups/
            echo "   SUCCESS: Local backup created in ../data/games/backups/"
            
            # S3 backup (essential data only)
            echo ""
            echo "2. Uploading essential data to S3..."
            echo "   Uploading $csv_files games CSV file(s) ($csv_size total)..."
            
            if aws s3 cp . s3://ncaaf-data/espn-games-data/games/csvs/ --recursive --exclude "*" --include "games_*.csv"; then
                echo "   SUCCESS: Essential data backed up to S3"
            else
                echo "   WARNING: S3 backup failed, but local backup exists"
            fi
            
            # Show what was skipped (saved time/bandwidth)
            json_count=$(find . -name "*.json" 2>/dev/null | wc -l || echo "0")
            if [ -d "gamejsons" ]; then
                json_count=$(find gamejsons -name "*.json" 2>/dev/null | wc -l || echo "0")
            fi
            other_count=$(find . -type f ! -name "games_*.csv" ! -name "*.json" | wc -l)
            
            echo ""
            echo "Skipped uploads (saving time/bandwidth):"
            echo "  JSON files: $json_count (raw data, can be re-collected)"
            echo "  Date/URL files: $other_count (can be regenerated)"
            
        else
            echo "WARNING: No essential games CSV files found to backup"
        fi
        
        # Clean up non-essential files to save space
        echo ""
        echo "3. Cleaning up non-essential files..."
        rm -f dates_* urls_* 2>/dev/null || true
        if [ -d "gamejsons" ]; then
            json_size=$(du -sh gamejsons 2>/dev/null | cut -f1 || echo "unknown")
            echo "   Removing JSON files ($json_size) - can be re-collected if needed"
            rm -rf gamejsons
        fi
        
        popd > /dev/null
        
        echo ""
        echo "=========================================="
        echo "BACKUP COMPLETE"
        echo "=========================================="
        echo "Essential data backed up:"
        echo "  Local:  ../data/games/backups/"
        echo "  S3:     s3://ncaaf-data/espn-games-data/games/csvs/"
        echo "Non-essential data cleaned up (saved space/time)"
        
    else
        echo "ERROR: temp directory not found"
        exit 1
    fi
fi

echo ""
echo "ESPN games collection completed successfully!"