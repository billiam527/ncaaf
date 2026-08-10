#!/usr/bin/env bash
set -e

echo "CFBD Spreads Collection Script"

# Parse arguments (handles both old and new format)
if [ $# -eq 4 ] && [ "$1" = "--start_year" ] && [ "$3" = "--end_year" ]; then
    # New format: --start_year 2023 --end_year 2024
    START_YEAR=$2
    END_YEAR=$4
    echo "Collecting data for years: $START_YEAR to $END_YEAR"
    
    # Run Python script with year arguments
    if python3 scrape_cfbd_data.py --start_year "$START_YEAR" --end_year "$END_YEAR"; then
        echo "SUCCESS: CFBD collection completed successfully"
    else
        echo "WARNING: CFBD collection had issues (may be normal)"
        # Don't exit 1 for CFBD issues
    fi
elif [ $# -eq 2 ]; then
    # Old format: START_YEAR END_YEAR
    START_YEAR=$1
    END_YEAR=$2
    echo "Collecting data for years: $START_YEAR to $END_YEAR"
    
    # Run Python script with year arguments
    if python3 scrape_cfbd_data.py --start_year "$START_YEAR" --end_year "$END_YEAR"; then
        echo "SUCCESS: CFBD collection completed successfully"
    else
        echo "WARNING: CFBD collection had issues (may be normal)"
    fi
else
    echo "Collecting data with default parameters (current year)"
    
    # Run Python script without arguments (uses defaults)
    if python3 scrape_cfbd_data.py; then
        echo "SUCCESS: CFBD collection completed successfully"
    else
        echo "WARNING: CFBD collection had issues (may be normal)"
    fi
fi

# Check if running from master script (batch upload mode)
if [ "$SKIP_S3_UPLOAD" = true ]; then
    echo ""
    echo "=========================================="
    echo "S3 UPLOAD DEFERRED TO BATCH"
    echo "=========================================="
    
    # Check if output file was created
    if [ -f "cfbd_spread_data.csv" ]; then
        file_size=$(wc -l < cfbd_spread_data.csv)
        echo "Files saved locally for batch upload:"
        echo "  cfbd_spread_data.csv ($file_size rows)"
        echo "  Location: $(pwd)"
        
        # Create local backup anyway (fast)
        mkdir -p ../data/cfbd/backups 2>/dev/null || true
        cp cfbd_spread_data.csv ../data/cfbd/backups/ 2>/dev/null || true
        echo "  Local backup: ../data/cfbd/backups/"
    else
        echo "WARNING: Expected output file 'cfbd_spread_data.csv' not found"
        echo "   This may be normal if no data was available for the specified years"
    fi
    
    echo "Batch upload will handle S3 transfer efficiently"
    
else
    # STANDALONE MODE: Do smart backup when script runs alone
    echo ""
    echo "=========================================="
    echo "SMART BACKUP - ESSENTIAL DATA ONLY"
    echo "=========================================="
    
    # Check if output file was created
    if [ -f "cfbd_spread_data.csv" ]; then
        file_size=$(wc -l < cfbd_spread_data.csv)
        echo "Essential file found:"
        echo "  cfbd_spread_data.csv ($file_size rows)"
        
        # Local backup first
        echo ""
        echo "1. Creating local backup..."
        mkdir -p ../data/cfbd/backups
        cp cfbd_spread_data.csv ../data/cfbd/backups/
        echo "   SUCCESS: Local backup created in ../data/cfbd/backups/"
        
        # S3 backup
        echo ""
        echo "2. Uploading to S3..."
        echo "   Uploading cfbd_spread_data.csv ($file_size rows)..."
        
        if aws s3 cp cfbd_spread_data.csv s3://ncaaf-data/cfbd-data/cfbd_spread_data.csv; then
            echo "   SUCCESS: Successfully uploaded to S3"
        else
            echo "   WARNING: S3 upload failed, but local backup exists"
            # Don't exit 1 - upload failure shouldn't stop pipeline
        fi
        
    else
        echo "WARNING: Expected output file 'cfbd_spread_data.csv' not found"
        echo "   This may be normal if no data was available for the specified years"
    fi
    
    echo ""
    echo "=========================================="
    echo "BACKUP COMPLETE"
    echo "=========================================="
fi

echo ""
echo "CFBD collection process completed!"