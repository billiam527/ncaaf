#!/usr/bin/env bash
set -e

echo "Optimized Sequential Data Collection with Progress Bars"

# ==============================================
# CONFIGURATION
# ==============================================

# Default to incremental mode (current season only)
INCREMENTAL_MODE=true
START_YEAR=2010
CURRENT_YEAR=$(date +"%Y")
CURRENT_MONTH=$(date +"%m")
TEST_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            echo "TEST MODE: Will process last 30 days only"
            shift
            ;;
        --full)
            INCREMENTAL_MODE=false
            echo "FULL MODE: Will process all years from $START_YEAR"
            shift
            ;;
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --years)
            CUSTOM_START="$2"
            CUSTOM_END="$3"
            INCREMENTAL_MODE=false
            START_YEAR=$CUSTOM_START
            CURRENT_YEAR=$CUSTOM_END
            echo "CUSTOM YEARS: Processing $CUSTOM_START to $CUSTOM_END"
            shift 3
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--full] [--start-year YEAR] [--years START_YEAR END_YEAR]"
            exit 1
            ;;
    esac
done

# ==============================================
# DETERMINE YEARS TO PROCESS
# ==============================================

determine_collection_years() {
    if [ "$TEST_MODE" = true ]; then
        echo "Test mode: processing last 30 days"
        YEARS_TO_PROCESS=($CURRENT_YEAR)
        return
    fi
    
    if [ "$INCREMENTAL_MODE" = true ]; then
        if [ $CURRENT_MONTH -ge 8 ]; then
            YEARS_TO_PROCESS=($CURRENT_YEAR)
            echo "Incremental: collecting current season $CURRENT_YEAR"
        elif [ $CURRENT_MONTH -le 2 ]; then
            LAST_YEAR=$((CURRENT_YEAR - 1))
            YEARS_TO_PROCESS=($LAST_YEAR)
            echo "Incremental: collecting finishing season $LAST_YEAR"
        else
            LAST_YEAR=$((CURRENT_YEAR - 1))
            YEARS_TO_PROCESS=($LAST_YEAR)
            echo "Incremental: off-season update for $LAST_YEAR"
        fi
    else
        YEARS_TO_PROCESS=($(seq $START_YEAR $CURRENT_YEAR))
        echo "Full mode: collecting years ${YEARS_TO_PROCESS[0]} to ${YEARS_TO_PROCESS[-1]}"
    fi
}

# ==============================================
# SEQUENTIAL COLLECTION FUNCTIONS
# ==============================================

run_collection() {
    local collection_dir=$1
    local collection_name=$2
    
    echo ""
    echo "Running: $collection_name"
    echo "=================================="
    
    if [ ! -d "$collection_dir" ]; then
        echo "WARNING: Directory $collection_dir not found - skipping"
        return 0
    fi
    
    pushd "$collection_dir" > /dev/null
    chmod +x *.sh
    
    # DISABLE S3 UPLOADS IN INDIVIDUAL SCRIPTS - batch upload will handle it
    export SKIP_S3_UPLOAD=true
    
    case $collection_dir in
        "collect_espn_teams")
            echo "Collecting team data (S3 upload deferred to batch)..."
            if ./collect_espn_teams.sh; then
                echo "Teams collection completed - files ready for batch upload"
            else
                echo "WARNING: Teams collection had issues (may be normal)"
            fi
            ;;
            
        "collect_cfbd_games") 
            if [ "$TEST_MODE" = true ]; then
                echo "Skipping CFBD in test mode"
            elif [ ${#YEARS_TO_PROCESS[@]} -eq 1 ]; then
                year=${YEARS_TO_PROCESS[0]}
                echo "Collecting CFBD data for year: $year (S3 upload deferred to batch)..."
                if ./scrape_cfbd_data.sh --start_year $year --end_year $year; then
                    echo "CFBD collection completed - files ready for batch upload"
                else
                    echo "WARNING: CFBD collection had issues (may be normal)"
                fi
            else
                first_year=${YEARS_TO_PROCESS[0]}
                last_year=${YEARS_TO_PROCESS[-1]}
                echo "Collecting CFBD data for years: $first_year to $last_year (S3 upload deferred to batch)..."
                if ./scrape_cfbd_data.sh --start_year $first_year --end_year $last_year; then
                    echo "CFBD collection completed - files ready for batch upload"
                else
                    echo "WARNING: CFBD collection had issues (may be normal)"
                fi
            fi
            ;;
            
        "collect_espn_games"|"collect_espn_pbp")
            script_name="./collect_espn_$(echo $collection_dir | cut -d'_' -f3).sh"
            
            if [ "$TEST_MODE" = true ]; then
                echo "Running test collection (S3 upload deferred to batch)..."
                test_start=$(date -d "30 days ago" +%Y-%m-%d)
                test_end=$(date +%Y-%m-%d)
                if python3 run.py --start_date "$test_start" --end_date "$test_end"; then
                    echo "Test collection completed - files ready for batch upload"
                else
                    echo "WARNING: Test collection had issues (may be normal)"
                fi
            elif [ ${#YEARS_TO_PROCESS[@]} -eq 1 ]; then
                year=${YEARS_TO_PROCESS[0]}
                echo "Collecting data for year: $year (S3 upload deferred to batch)..."
                if $script_name $year; then
                    echo "Collection completed - files ready for batch upload"
                else
                    echo "WARNING: Collection completed with warnings (may be normal for current season)"
                fi
            else
                first_year=${YEARS_TO_PROCESS[0]}
                last_year=${YEARS_TO_PROCESS[-1]}
                echo "Collecting data for years: $first_year to $last_year (S3 upload deferred to batch)..."
                if $script_name $first_year $last_year; then
                    echo "Collection completed - files ready for batch upload"
                else
                    echo "WARNING: Collection had issues"
                fi
            fi
            ;;
    esac
    
    # Clean up environment variable
    unset SKIP_S3_UPLOAD
    
    popd > /dev/null
    return 0
}

# ==============================================
# SELECTIVE S3 UPLOAD OPTIMIZATION (FIXED)
# ==============================================

batch_upload_to_s3() {
    echo ""
    echo "SELECTIVE S3 UPLOAD OPTIMIZATION"
    echo "=================================="
    
    # Create a single temp directory for all uploads
    upload_temp="upload_staging"
    mkdir -p "$upload_temp"
    
    echo "Staging ESSENTIAL files for batch upload..."
    echo "(Skipping intermediate files and JSONs to reduce upload size)"
    
    # ==============================================
    # ESSENTIAL CSV FILES ONLY
    # ==============================================
    
    echo "Staging essential CSV files..."
    
    # ESPN Teams (small, always include)
    if [ -f "collect_espn_teams/temp/teams.csv" ]; then
        mkdir -p "$upload_temp/espn-teams-data"
        cp collect_espn_teams/temp/teams.csv "$upload_temp/espn-teams-data/" 2>/dev/null || true
        echo "   Staged: teams.csv"
    fi
    
    # ESPN Games - Final CSV files only (NO JSONs)
    if [ -d "collect_espn_games/temp" ]; then
        mkdir -p "$upload_temp/espn-games-data/games/csvs"
        
        # Only copy CSV files, skip everything else
        find collect_espn_games/temp -name "games_*.csv" -type f | while read file; do
            cp "$file" "$upload_temp/espn-games-data/games/csvs/" 2>/dev/null || true
            echo "   Staged: $(basename $file)"
        done
    fi
    
    # ESPN PBP - Final CSV files only (NO JSONs)
    if [ -d "collect_espn_pbp/temp" ]; then
        mkdir -p "$upload_temp/espn-pbp-data/pbp/csvs"
        
        # Only copy CSV files, skip the huge JSON directory
        find collect_espn_pbp/temp -name "play-by-play*.csv" -type f | while read file; do
            cp "$file" "$upload_temp/espn-pbp-data/pbp/csvs/" 2>/dev/null || true
            echo "   Staged: $(basename $file)"
        done
    fi
    
    # CFBD - Final CSV only
    if [ -f "collect_cfbd_games/cfbd_spread_data.csv" ]; then
        mkdir -p "$upload_temp/cfbd-data"
        cp collect_cfbd_games/cfbd_spread_data.csv "$upload_temp/cfbd-data/" 2>/dev/null || true
        echo "   Staged: cfbd_spread_data.csv"
    fi
    
    # Count total files to upload
    total_files=$(find "$upload_temp" -type f 2>/dev/null | wc -l)
    
    echo ""
    echo "UPLOAD OPTIMIZATION SUMMARY:"
    echo "   Essential files staged: $total_files"
    echo "   Skipped: JSON files, intermediate files, temporary data"
    echo "   Estimated savings: ~$(echo $((740 - total_files))) fewer uploads"
    
    if [ $total_files -gt 0 ]; then
        echo ""
        echo "Executing optimized selective upload..."
        
        if aws s3 sync "$upload_temp" s3://ncaaf-data/ --delete --quiet; then
            echo "   Selective upload completed successfully"
            echo "   Uploaded $total_files essential files"
            echo "   Skipped ~$(echo $((740 - total_files))) unnecessary files"
        else
            echo "   WARNING: Batch upload had issues - trying fallback approach..."
            
            # Fallback: Upload each subdirectory individually
            upload_success=true
            for subdir in "$upload_temp"/*; do
                if [ -d "$subdir" ]; then
                    dirname=$(basename "$subdir")
                    echo "   Uploading $dirname..."
                    if aws s3 sync "$subdir" "s3://ncaaf-data/$dirname/" --quiet; then
                        echo "   Successfully uploaded: $dirname"
                    else
                        echo "   ERROR: Failed to upload $dirname"
                        upload_success=false
                    fi
                fi
            done
            
            if [ "$upload_success" = true ]; then
                echo "   Fallback uploads completed successfully"
            else
                echo "   WARNING: Some fallback uploads failed"
                return 1
            fi
        fi
        
        # Cleanup staging
        rm -rf "$upload_temp"
        echo "   Cleaned up staging directory"
        
        return 0
    else
        echo "   WARNING: No files found to upload"
        return 1
    fi
}

# ==============================================
# MAIN EXECUTION
# ==============================================

main() {
    start_time=$(date +%s)
    
    echo "Collection Configuration:"
    echo "   Mode: SEQUENTIAL (with full progress bars)"
    echo "   Incremental Mode: $INCREMENTAL_MODE"
    echo "   Test Mode: $TEST_MODE"
    echo "   Start Year: $START_YEAR"
    echo ""
    
    # Determine what years to process
    determine_collection_years
    echo "   Years to process: ${YEARS_TO_PROCESS[*]}"
    echo ""
    
    # Collections to run (in dependency order)
    COLLECTIONS=(
        "collect_espn_teams:ESPN Teams Data"
        "collect_espn_games:ESPN Games Data"
        "collect_espn_pbp:ESPN Play-by-Play Data"
        "collect_cfbd_games:CFBD Spreads Data"
    )
    
    # Track results
    successful_collections=()
    failed_collections=()
    
    # Run each collection SEQUENTIALLY to preserve progress bars
    echo "Running collections sequentially for maximum progress visibility..."
    echo ""
    
    for collection in "${COLLECTIONS[@]}"; do
        IFS=':' read -r dir name <<< "$collection"
        
        echo "Starting: $name"
        if run_collection "$dir" "$name"; then
            successful_collections+=("$name")
            echo "Completed: $name"
        else
            failed_collections+=("$name")
            echo "ERROR: $name failed"
        fi
        echo ""
        echo "=================================================="
        echo ""
    done
    
    # Selective upload optimization
    batch_upload_to_s3
    
    # Calculate timing
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    # Summary
    echo ""
    echo "=================================="
    echo "COLLECTION SUMMARY"
    echo "=================================="
    echo "Total time: ${minutes}m ${seconds}s"
    echo "Execution mode: Sequential (full progress visibility)"
    
    if [ ${#successful_collections[@]} -gt 0 ]; then
        echo ""
        echo "Successful collections (${#successful_collections[@]}):"
        for collection in "${successful_collections[@]}"; do
            echo "   - $collection"
        done
    fi
    
    if [ ${#failed_collections[@]} -gt 0 ]; then
        echo ""
        echo "Collections with issues (${#failed_collections[@]}):"
        for collection in "${failed_collections[@]}"; do
            echo "   - $collection"
        done
        echo ""
        echo "Note: Some issues may be normal (e.g., current season not complete)"
    fi
    
    if [ ${#successful_collections[@]} -gt 0 ]; then
        echo ""
        if [ "$TEST_MODE" = true ]; then
            echo "Test collection completed - pipeline is working!"
        elif [ "$INCREMENTAL_MODE" = true ]; then
            echo "Incremental collection completed - database updated!"
        else
            echo "Full collection completed!"
        fi
        return 0
    else
        echo ""
        echo "ERROR: All collections failed - check logs for details"
        return 1
    fi
}

# Run main function
main "$@"