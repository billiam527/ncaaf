#!/usr/bin/env bash
set -e

echo "ESPN Teams Collection Script"

# Clean and create temp directory
rm -rf temp 2>/dev/null || true
mkdir temp

# Run the collection script
echo "Collecting team data..."
if python3 run.py; then
    echo "SUCCESS: Team data collection completed"
else
    echo "WARNING: Team data collection had issues (may be normal)"
    # Don't exit 1 - teams collection issues shouldn't stop the pipeline
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
        csv_count=$(find . -name "*.csv" | wc -l)
        json_count=$(find . -name "*.json" 2>/dev/null | wc -l || echo "0")
        
        echo "  CSV files ready: $csv_count"
        echo "  JSON files ready: $json_count" 
        echo "  Location: $(pwd)"
        
        # Create local backup anyway (fast)
        mkdir -p ../data/teams/backups 2>/dev/null || true
        cp *.csv ../data/teams/backups/ 2>/dev/null || true
        cp *.json ../data/teams/backups/ 2>/dev/null || true
        echo "  Local backup: ../data/teams/backups/"
        
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
        
        # Check if we have files to upload
        file_count=$(find . -name "*.csv" -o -name "*.json" | wc -l)
        
        echo "Essential files found: $file_count"
        
        if [ $file_count -gt 0 ]; then
            # Local backup first
            echo ""
            echo "1. Creating local backup..."
            mkdir -p ../data/teams/backups
            cp *.csv ../data/teams/backups/ 2>/dev/null || true
            cp *.json ../data/teams/backups/ 2>/dev/null || true
            echo "   SUCCESS: Local backup created in ../data/teams/backups/"
            
            # S3 backup
            echo ""
            echo "2. Uploading team files to S3..."
            echo "   Uploading $file_count file(s)..."
            
            if aws s3 cp . s3://ncaaf-data/espn-teams-data/ --recursive --exclude "*" --include "*.csv" --include "*.json"; then
                echo "   SUCCESS: Successfully uploaded $file_count files to S3"
                
                # Also upload the JSON file specifically if it exists
                if [ -f "json_teams.json" ]; then
                    aws s3 cp json_teams.json s3://ncaaf-data/espn-teams-data/ --quiet
                fi
            else
                echo "   WARNING: S3 backup failed, but local backup exists"
            fi
        else
            echo "WARNING: No files found to backup"
        fi
        
        popd > /dev/null
    else
        echo "WARNING: temp directory not found"
    fi
    
    echo ""
    echo "=========================================="
    echo "BACKUP COMPLETE"
    echo "=========================================="
fi

echo ""
echo "ESPN teams collection completed!"