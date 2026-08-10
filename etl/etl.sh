#!/usr/bin/env bash
set -e

echo "Optimized College Football Data Pipeline"

# ==============================================
# CONFIGURATION - EDIT THESE TO CONTROL PIPELINE
# ==============================================

# Year control - set what years to process
INCREMENTAL_MODE=true  # Set to false to reprocess all years from START_YEAR
START_YEAR=2010        # Only used if INCREMENTAL_MODE=false
CURRENT_YEAR=$(date +"%Y")
CURRENT_MONTH=$(date +"%m")

# Test mode - processes only last 30 days for quick testing
TEST_MODE=false        # Set to true for quick testing
TEST_DAYS=30          # Number of recent days to test with

# Pipeline steps to run
SKIP_COLLECTION=false  # Set to true to skip data collection
SKIP_FORMATTING=false # Set to true to skip data formatting  
SKIP_SUMMARIZATION=false # Set to true to skip summarization
SKIP_UPLOAD=false     # Set to true to skip final S3 upload

# Safety floor: refuse to sync to S3 if fewer than this many files were staged.
# Guards against a failed collect/format run publishing an empty tree.
MIN_STAGED_FILES=3

# Set to true only once summarize.sh actually succeeds this run. Gates whether
# summarize/results/*.csv gets republished to s3://ncaaf-data/model_data/.
SUMMARIZE_SUCCEEDED=false

# Custom year range variables
CUSTOM_START=""
CUSTOM_END=""

# ==============================================
# YEAR LOGIC
# ==============================================

determine_years_to_process() {
    if [ "$TEST_MODE" = true ]; then
        # Test mode: just last X days
        test_start=$(date -d "$TEST_DAYS days ago" +%Y-%m-%d)
        test_end=$(date +%Y-%m-%d)
        echo "TEST MODE: Processing $TEST_DAYS days ($test_start to $test_end)"
        YEARS_TO_PROCESS=($CURRENT_YEAR)
        CUSTOM_DATE_RANGE="--start_date $test_start --end_date $test_end"
        return
    fi
    
    # Handle custom year range
    if [ -n "$CUSTOM_START" ] && [ -n "$CUSTOM_END" ]; then
        YEARS_TO_PROCESS=($(seq $CUSTOM_START $CUSTOM_END))
        echo "CUSTOM RANGE: Processing years $CUSTOM_START to $CUSTOM_END"
        CUSTOM_DATE_RANGE=""
        return
    fi
    
    if [ "$INCREMENTAL_MODE" = true ]; then
        # Incremental: only process current season
        if [ $CURRENT_MONTH -ge 8 ]; then
            # Season in progress (Aug-Dec)
            YEARS_TO_PROCESS=($CURRENT_YEAR)
            echo "INCREMENTAL MODE: Processing current season $CURRENT_YEAR"
        elif [ $CURRENT_MONTH -le 2 ]; then
            # Season finishing (Jan-Feb)
            LAST_YEAR=$((CURRENT_YEAR - 1))
            YEARS_TO_PROCESS=($LAST_YEAR)
            echo "INCREMENTAL MODE: Processing finishing season $LAST_YEAR"
        else
            # Off-season (Mar-Jul) - check for any missed data from last season
            LAST_YEAR=$((CURRENT_YEAR - 1))
            YEARS_TO_PROCESS=($LAST_YEAR)
            echo "INCREMENTAL MODE: Off-season cleanup for $LAST_YEAR"
        fi
    else
        # Full mode: process all years from START_YEAR
        YEARS_TO_PROCESS=($(seq $START_YEAR $CURRENT_YEAR))
        echo "FULL MODE: Processing all years from $START_YEAR to $CURRENT_YEAR"
    fi
    
    CUSTOM_DATE_RANGE=""
}

# ==============================================
# FIXED GLOBAL S3 BATCH OPTIMIZATION FUNCTION
# ==============================================

optimize_s3_operations() {
    echo ""
    echo "SELECTIVE S3 OPTIMIZATION"
    echo "=================================="
    
    # Create global staging directory
    GLOBAL_STAGING="staging_for_s3"
    rm -rf "$GLOBAL_STAGING" 2>/dev/null || true
    mkdir -p "$GLOBAL_STAGING"
    
    echo "Collecting ESSENTIAL pipeline outputs for batch upload..."
    echo "(Skipping intermediate files and JSONs to reduce upload size)"
    
    # Collect only essential files
    local files_staged=0
    
    # ==============================================
    # ESSENTIAL CSV FILES ONLY
    # ==============================================
    
    # ESPN Teams (small, always include)
    if [ -f "collect/collect_espn_teams/temp/teams.csv" ]; then
        mkdir -p "$GLOBAL_STAGING/espn-teams-data"
        cp collect/collect_espn_teams/temp/teams.csv "$GLOBAL_STAGING/espn-teams-data/" 2>/dev/null || true
        echo "  Staged: teams.csv"
    fi
    
    # ESPN Games - Final CSV only
    if [ -f "collect/collect_espn_games/temp/games_"*".csv" ]; then
        mkdir -p "$GLOBAL_STAGING/espn-games-data/games/csvs"
        # Copy individual year files
        find collect/collect_espn_games/temp -name "games_*.csv" | while read file; do
            cp "$file" "$GLOBAL_STAGING/espn-games-data/games/csvs/" 2>/dev/null || true
        done
        echo "  Staged: ESPN games CSV files"
    fi
    
    # ESPN PBP - Final CSV only  
    if [ -f "collect/collect_espn_pbp/temp/play-by-play"*".csv" ]; then
        mkdir -p "$GLOBAL_STAGING/espn-pbp-data/pbp/csvs"
        # Copy individual year files
        find collect/collect_espn_pbp/temp -name "play-by-play*.csv" | while read file; do
            cp "$file" "$GLOBAL_STAGING/espn-pbp-data/pbp/csvs/" 2>/dev/null || true
        done
        echo "  Staged: ESPN PBP CSV files"
    fi
    
    # CFBD - Final CSV only
    if [ -f "collect/collect_cfbd_games/cfbd_spread_data.csv" ]; then
        mkdir -p "$GLOBAL_STAGING/cfbd-data"
        cp collect/collect_cfbd_games/cfbd_spread_data.csv "$GLOBAL_STAGING/cfbd-data/" 2>/dev/null || true
        echo "  Staged: CFBD spreads CSV"
    fi
    
    # ==============================================
    # FORMATTED/PROCESSED FILES
    # ==============================================
    
    # Format outputs - Final processed files only
    if [ -f "format/format_espn_games/temp/games.csv" ]; then
        mkdir -p "$GLOBAL_STAGING/espn-games-data/games/csvs"
        cp format/format_espn_games/temp/games.csv "$GLOBAL_STAGING/espn-games-data/games/csvs/games_combined.csv" 2>/dev/null || true
        echo "  Staged: Combined games CSV"
    fi
    
    if [ -f "format/format_espn_pbp/temp/pbp_edit.csv" ]; then
        mkdir -p "$GLOBAL_STAGING/espn-pbp-data/pbp/csvs"
        cp format/format_espn_pbp/temp/pbp_edit.csv "$GLOBAL_STAGING/espn-pbp-data/pbp/csvs/pbp_enhanced.csv" 2>/dev/null || true
        echo "  Staged: Enhanced PBP CSV"
    fi
    
    # ==============================================
    # FINAL ANALYSIS RESULTS
    # ==============================================
    
    # Summarize outputs - Final analysis files.
    # Only stage these if summarization actually ran and succeeded this pass;
    # otherwise we republish stale results as though they were fresh.
    if [ "$SUMMARIZE_SUCCEEDED" != true ]; then
        echo "  Skipped: analysis results (summarization did not succeed this run)"
    elif [ -d "summarize/results" ]; then
        mkdir -p "$GLOBAL_STAGING/model_data"
        find summarize/results -name "*.csv" | while read file; do
            cp "$file" "$GLOBAL_STAGING/model_data/" 2>/dev/null || true
        done
        echo "  Staged: Analysis results"
    fi
    
    # ==============================================
    # OPTIONAL: COMPRESSED JSON ARCHIVE
    # ==============================================
    
    # Optionally create compressed archive of JSON files instead of uploading individually
    json_files_found=0
    
    # Check for JSON files
    if [ -d "collect/collect_espn_pbp/temp/pbpjsons" ]; then
        json_count=$(find collect/collect_espn_pbp/temp/pbpjsons -name "*.json" | wc -l)
        if [ $json_count -gt 0 ]; then
            echo "  Found $json_count PBP JSON files - creating compressed archive..."
            mkdir -p "$GLOBAL_STAGING/espn-pbp-data/archives"
            
            # Create compressed archive instead of individual files
            tar -czf "$GLOBAL_STAGING/espn-pbp-data/archives/pbp_jsons_$(date +%Y%m%d).tar.gz" \
                -C collect/collect_espn_pbp/temp pbpjsons/ 2>/dev/null || true
            
            echo "  Staged: PBP JSONs archive (compressed)"
            json_files_found=$((json_files_found + json_count))
        fi
    fi
    
    if [ -d "collect/collect_espn_games/temp/gamejsons" ]; then
        json_count=$(find collect/collect_espn_games/temp/gamejsons -name "*.json" | wc -l)
        if [ $json_count -gt 0 ]; then
            echo "  Found $json_count Games JSON files - creating compressed archive..."
            mkdir -p "$GLOBAL_STAGING/espn-games-data/archives"
            
            # Create compressed archive instead of individual files  
            tar -czf "$GLOBAL_STAGING/espn-games-data/archives/games_jsons_$(date +%Y%m%d).tar.gz" \
                -C collect/collect_espn_games/temp gamejsons/ 2>/dev/null || true
            
            echo "  Staged: Games JSONs archive (compressed)"
            json_files_found=$((json_files_found + json_count))
        fi
    fi
    
    # Count total files to upload
    files_staged=$(find "$GLOBAL_STAGING" -type f 2>/dev/null | wc -l)
    
    echo ""
    echo "UPLOAD OPTIMIZATION SUMMARY:"
    echo "  Essential CSV files: ~$((files_staged - (json_files_found > 0 ? 2 : 0)))"
    if [ $json_files_found -gt 0 ]; then
        echo "  JSON archives: 2 compressed files (instead of $json_files_found individual)"
    fi
    echo "  Total files to upload: $files_staged (was ~750+ before optimization)"
    
    # Refuse to publish a near-empty staging tree. A failed collect/format run
    # leaves almost nothing staged, and syncing that up overwrites good data.
    if [ $files_staged -lt $MIN_STAGED_FILES ]; then
        echo ""
        echo "  ERROR: Only $files_staged file(s) staged (minimum $MIN_STAGED_FILES)."
        echo "  Refusing to upload - this looks like a failed pipeline run."
        echo "  Staged contents:"
        find "$GLOBAL_STAGING" -type f 2>/dev/null | sed 's/^/    /'
        echo "  Leaving $GLOBAL_STAGING/ in place for inspection."
        return 1
    fi

    if [ $files_staged -gt 0 ]; then
        echo ""
        echo "Executing optimized selective upload..."

        upload_start=$(date +%s)

        # NOTE: deliberately no --delete. It previously wiped the bucket's raw
        # data whenever a run staged fewer files than the bucket already held.
        if aws s3 sync "$GLOBAL_STAGING" s3://ncaaf-data/ --quiet; then
            upload_end=$(date +%s)
            upload_duration=$((upload_end - upload_start))
            
            echo "  SUCCESS: Selective upload completed in ${upload_duration}s"
            echo "  Uploaded $files_staged essential files"
            echo "  Saved ~$((750 - files_staged)) unnecessary uploads"
            echo "  Upload efficiency: $(($files_staged / ($upload_duration + 1))) files/second"
        else
            echo "  WARNING: Batch upload had issues - trying fallback approach..."
            
            # Fallback: Upload each subdirectory individually
            upload_success=true
            for subdir in "$GLOBAL_STAGING"/*; do
                if [ -d "$subdir" ]; then
                    dirname=$(basename "$subdir")
                    echo "  Uploading $dirname..."
                    if aws s3 sync "$subdir" "s3://ncaaf-data/$dirname/" --quiet; then
                        echo "  SUCCESS: $dirname uploaded"
                    else
                        echo "  ERROR: $dirname upload failed"
                        upload_success=false
                    fi
                fi
            done
            
            if [ "$upload_success" = true ]; then
                echo "  SUCCESS: Fallback uploads completed"
            else
                echo "  WARNING: Some fallback uploads failed"
                return 1
            fi
        fi
        
        # Cleanup staging
        rm -rf "$GLOBAL_STAGING"
        echo "  Cleaned up staging directory"
        
        return 0
    else
        echo "  WARNING: No files found to upload"
        return 1
    fi
}

# ==============================================
# OPTIMIZED COLLECTION FUNCTIONS
# ==============================================

run_optimized_collection() {
    echo "=================================="
    echo "STEP 1: Optimized Data Collection"
    echo "=================================="
    
    if [ "$SKIP_COLLECTION" = true ]; then
        echo "Skipping collection (SKIP_COLLECTION=true)"
        return 0
    fi
    
    # Run the optimized collection script
    if [ -d "collect" ]; then
        pushd "collect" > /dev/null
        chmod +x *.sh
        
        # Pass parameters to collection script
        collection_args=""
        if [ "$TEST_MODE" = true ]; then
            collection_args="--test"
        elif [ "$INCREMENTAL_MODE" = false ]; then
            if [ -n "$CUSTOM_START" ] && [ -n "$CUSTOM_END" ]; then
                collection_args="--years $CUSTOM_START $CUSTOM_END"
            else
                collection_args="--full --start-year $START_YEAR"
            fi
        fi
        
        if ./collect.sh $collection_args; then
            echo "SUCCESS: Collection completed"
            popd > /dev/null
            return 0
        else
            echo "WARNING: Collection completed with some issues (may be normal)"
            popd > /dev/null
            return 0  # Don't fail pipeline for collection warnings
        fi
    else
        echo "ERROR: collect directory not found"
        return 1
    fi
}

# ==============================================
# ZERO-COPY FORMATTING (NO S3 ROUNDTRIP)
# ==============================================

run_zero_copy_formatting() {
    echo ""
    echo "=================================="
    echo "STEP 2: Zero-Copy Data Formatting"
    echo "=================================="
    
    if [ "$SKIP_FORMATTING" = true ]; then
        echo "Skipping formatting (SKIP_FORMATTING=true)"
        return 0
    fi
    
    if [ -d "format" ]; then
        pushd "format" > /dev/null
        chmod +x *.sh
        
        # Determine format arguments
        if [ "$TEST_MODE" = true ]; then
            echo "Skipping formatting in test mode"
        else
            # Pass year parameters to formatting
            if [ -n "$CUSTOM_START" ] && [ -n "$CUSTOM_END" ]; then
                format_args="$CUSTOM_START $CUSTOM_END"
            else
                determine_years_to_process
                if [ ${#YEARS_TO_PROCESS[@]} -eq 1 ]; then
                    format_args="${YEARS_TO_PROCESS[0]}"
                else
                    first_year=${YEARS_TO_PROCESS[0]}
                    last_year=${YEARS_TO_PROCESS[-1]}
                    format_args="$first_year $last_year"
                fi
            fi
            
            if ./format.sh $format_args; then
                echo "SUCCESS: Formatting completed"
            else
                echo "WARNING: Formatting had some issues"
            fi
        fi
        
        popd > /dev/null
    else
        echo "WARNING: format directory not found - skipping"
    fi
    
    return 0
}

# ==============================================
# SMART SUMMARIZATION
# ==============================================

run_smart_summarization() {
    echo ""
    echo "=================================="
    echo "STEP 3: Smart Summarization"
    echo "=================================="
    
    if [ "$SKIP_SUMMARIZATION" = true ]; then
        echo "Skipping summarization (SKIP_SUMMARIZATION=true)"
        return 0
    fi
    
    if [ "$TEST_MODE" = true ]; then
        echo "Skipping summarization in test mode"
        return 0
    fi
    
    if [ -d "summarize" ]; then
        pushd "summarize" > /dev/null
        chmod +x *.sh
        
        if [ "$INCREMENTAL_MODE" = true ]; then
            echo "Running incremental summarization..."
        else
            echo "Running full summarization..."
        fi
        
        if ./summarize.sh; then
            echo "SUCCESS: Summarization completed"
            SUMMARIZE_SUCCEEDED=true
        else
            echo "WARNING: Summarization had issues"
            SUMMARIZE_SUCCEEDED=false
        fi
        
        popd > /dev/null
    else
        echo "WARNING: summarize directory not found - skipping"
    fi
    
    return 0
}

# ==============================================
# MAIN PIPELINE EXECUTION
# ==============================================

main() {
    # Track timing
    start_time=$(date +%s)
    
    echo "College Football Data Pipeline"
    echo "Started at: $(date)"
    echo ""
    
    # Print configuration
    echo "PIPELINE CONFIGURATION:"
    echo "  Incremental Mode: $INCREMENTAL_MODE"
    echo "  Test Mode: $TEST_MODE"
    echo "  Start Year: $START_YEAR"
    if [ -n "$CUSTOM_START" ] && [ -n "$CUSTOM_END" ]; then
        echo "  Custom Range: $CUSTOM_START to $CUSTOM_END"
    fi
    echo "  Skip Collection: $SKIP_COLLECTION"
    echo "  Skip Formatting: $SKIP_FORMATTING" 
    echo "  Skip Summarization: $SKIP_SUMMARIZATION"
    echo "  Skip Upload: $SKIP_UPLOAD"
    echo ""
    
    # Run pipeline steps
    step_success=true
    
    if run_optimized_collection; then
        echo "SUCCESS: Collection phase completed"
    else
        echo "ERROR: Collection phase failed"
        step_success=false
    fi
    
    if [ "$step_success" = true ]; then
        if run_zero_copy_formatting; then
            echo "SUCCESS: Formatting phase completed"
        else
            echo "ERROR: Formatting phase failed"
            step_success=false
        fi
    fi
    
    if [ "$step_success" = true ]; then
        if run_smart_summarization; then
            echo "SUCCESS: Summarization phase completed"
        else
            echo "ERROR: Summarization phase failed" 
            step_success=false
        fi
    fi
    
    # Global S3 batch optimization after all processing
    if [ "$step_success" = true ] && [ "$SKIP_UPLOAD" != true ]; then
        echo ""
        echo "FINAL OPTIMIZATION: Global S3 batch upload"
        if optimize_s3_operations; then
            echo "SUCCESS: Global S3 optimization completed"
        else
            echo "WARNING: Global S3 optimization had issues"
        fi
    fi
    
    # Final summary
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo ""
    echo "=================================="
    echo "PIPELINE SUMMARY"
    echo "=================================="
    echo "Completed at: $(date)"
    echo "Total time: ${minutes}m ${seconds}s"
    
    if [ "$step_success" = true ]; then
        echo "SUCCESS: All pipeline steps completed!"
        
        if [ "$TEST_MODE" = true ]; then
            echo ""
            echo "TEST MODE RESULTS:"
            echo "  Pipeline is working correctly"
            echo "  Set TEST_MODE=false for full execution"
        elif [ -n "$CUSTOM_START" ] && [ -n "$CUSTOM_END" ]; then
            echo ""
            echo "CUSTOM RANGE COMPLETE:"
            echo "  Years processed: $CUSTOM_START to $CUSTOM_END"
            echo "  Database updated with specified range"
        elif [ "$INCREMENTAL_MODE" = true ]; then
            echo ""
            echo "INCREMENTAL UPDATE COMPLETE:"
            determine_years_to_process
            echo "  Years processed: ${YEARS_TO_PROCESS[*]}"
            echo "  Database updated with latest data"
        fi
    else
        echo "ERROR: Pipeline failed - check logs above for details"
        exit 1
    fi
}

# ==============================================
# EXECUTION
# ==============================================

# Allow override via command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --full)
            INCREMENTAL_MODE=false
            shift
            ;;
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --years)
            # Custom year range: --years 2020 2024
            CUSTOM_START="$2"
            CUSTOM_END="$3"
            INCREMENTAL_MODE=false
            echo "CUSTOM YEARS: Processing $CUSTOM_START to $CUSTOM_END"
            shift 3
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
            shift
            ;;
        --skip-formatting)
            SKIP_FORMATTING=true
            shift
            ;;
        --skip-summarization) 
            SKIP_SUMMARIZATION=true
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--full] [--start-year YEAR] [--years START_YEAR END_YEAR] [--skip-collection] [--skip-formatting] [--skip-summarization] [--skip-upload]"
            exit 1
            ;;
    esac
done

# Run the main pipeline
main