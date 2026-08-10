#!/usr/bin/env bash

# Enhanced error tracking with detailed line reporting
set -eE
trap 'handle_error ${LINENO} $? "${BASH_COMMAND}" "${FUNCNAME[*]}"' ERR

# Global error tracking
SCRIPT_NAME="format.sh"
TOTAL_ERRORS=0
declare -a ERROR_LOG

# Enhanced error handler with context
handle_error() {
    local line_no=$1
    local exit_code=$2
    local failed_command=$3
    local function_stack=$4
    
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    
    echo "=================================================" >&2
    echo "ERROR #${TOTAL_ERRORS} in ${SCRIPT_NAME}" >&2
    echo "=================================================" >&2
    echo "Location: Line ${line_no}" >&2
    echo "Exit Code: ${exit_code}" >&2
    echo "Failed Command: ${failed_command}" >&2
    
    if [ -n "$function_stack" ] && [ "$function_stack" != "main" ]; then
        echo "Function Stack: ${function_stack}" >&2
    fi
    
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >&2
    
    # Context around the error
    if [ -f "$0" ]; then
        echo "Code Context:" >&2
        local start_line=$((line_no - 3))
        local end_line=$((line_no + 3))
        [ $start_line -lt 1 ] && start_line=1
        
        sed -n "${start_line},${end_line}p" "$0" | \
        awk -v target="$line_no" -v start="$start_line" \
        'NR==target-start+1 {print ">>>" NR+start-1 ":", $0; next} {print "   " NR+start-1 ":", $0}' >&2
    fi
    
    # Log error for summary
    ERROR_LOG+=("Line ${line_no}: ${failed_command} (exit ${exit_code})")
    
    echo "=================================================" >&2
    echo ""
}

# Validation functions with detailed error reporting
validate_directory() {
    local dir=$1
    local context=$2
    local line_caller=${BASH_LINENO[0]}
    
    echo "VALIDATION: Checking directory '$dir' ($context)"
    
    if [ ! -d "$dir" ]; then
        echo "ERROR at line $line_caller: Directory '$dir' not found ($context)" >&2
        echo "Current working directory: $(pwd)" >&2
        echo "Available directories:" >&2
        ls -la . | grep "^d" >&2 || echo "No directories found" >&2
        return 1
    fi
    
    echo "   SUCCESS: Directory '$dir' exists"
    return 0
}

validate_file() {
    local file=$1
    local context=$2
    local line_caller=${BASH_LINENO[0]}
    
    echo "VALIDATION: Checking file '$file' ($context)"
    
    if [ ! -f "$file" ]; then
        echo "ERROR at line $line_caller: File '$file' not found ($context)" >&2
        echo "Directory contents:" >&2
        ls -la "$(dirname "$file")" >&2 || echo "Directory doesn't exist" >&2
        return 1
    fi
    
    echo "   SUCCESS: File '$file' exists"
    return 0
}

validate_executable() {
    local script=$1
    local context=$2
    local line_caller=${BASH_LINENO[0]}
    
    echo "VALIDATION: Checking executable '$script' ($context)"
    
    if [ ! -f "$script" ]; then
        echo "ERROR at line $line_caller: Script '$script' not found ($context)" >&2
        return 1
    fi
    
    if [ ! -x "$script" ]; then
        echo "WARNING at line $line_caller: Script '$script' not executable, attempting chmod" >&2
        chmod +x "$script" || {
            echo "ERROR at line $line_caller: Could not make '$script' executable" >&2
            return 1
        }
    fi
    
    echo "   SUCCESS: Script '$script' is executable"
    return 0
}

echo "Optimized Data Formatting Pipeline with Expected Points Model"

# ==============================================
# CONFIGURATION
# ==============================================

START_YEAR=2010
END_YEAR=$(date +"%Y")
ENABLE_ZERO_COPY=true
BATCH_SIZE=100000  # Process large files in chunks
FORCE_EP_REGENERATE=${FORCE_EP_REGENERATE:-true}  # Default to true, can be overridden

# Parse arguments
if [ $# -eq 1 ]; then
    START_YEAR=$1
    END_YEAR=$((START_YEAR + 1))
    echo "Formatting single season: $START_YEAR-$END_YEAR"
elif [ $# -eq 2 ]; then
    START_YEAR=$1
    END_YEAR=$2
    echo "Formatting year range: $START_YEAR-$END_YEAR"
else
    echo "Formatting data from $START_YEAR to $END_YEAR"
fi

# ==============================================
# ZERO-COPY FILE DISCOVERY
# ==============================================

discover_local_files() {
    echo "ZERO-COPY OPTIMIZATION: Discovering local files..."
    
    declare -A local_file_map
    local_files_found=0
    
    # Search patterns for collection output files
    local search_paths=(
        "../collect/collect_espn_games/temp"
        "../collect/collect_espn_pbp/temp"
        "../../collect/collect_espn_games/temp"
        "../../collect/collect_espn_pbp/temp"
        "../collect_espn_games/temp"
        "../collect_espn_pbp/temp"
    )
    
    for search_path in "${search_paths[@]}"; do
        if [ -d "$search_path" ]; then
            echo "   Scanning: $search_path"
            
            # Find games files
            for file in "$search_path"/games_*.csv; do
                if [ -f "$file" ]; then
                    basename_file=$(basename "$file")
                    local_file_map["games:$basename_file"]="$file"
                    local_files_found=$((local_files_found + 1))
                    echo "   Found games: $basename_file"
                fi
            done
            
            # Find PBP files
            for file in "$search_path"/play-by-play_*.csv; do
                if [ -f "$file" ]; then
                    basename_file=$(basename "$file")
                    local_file_map["pbp:$basename_file"]="$file"
                    local_files_found=$((local_files_found + 1))
                    echo "   Found PBP: $basename_file"
                fi
            done
        fi
    done
    
    echo "   SUCCESS: Local file discovery complete: $local_files_found files found"
    
    # Export the associative array for use by formatters
    mkdir -p temp || {
        echo "ERROR at line ${LINENO}: Could not create temp directory" >&2
        return 1
    }
    
    for key in "${!local_file_map[@]}"; do
        echo "$key=${local_file_map[$key]}" >> temp/local_files.map
    done || {
        echo "ERROR at line ${LINENO}: Could not write to temp/local_files.map" >&2
        return 1
    }
    
    return 0  # Always return success
}

# ==============================================
# PARALLEL FORMATTING EXECUTION
# ==============================================

run_formatter_parallel() {
    local formatter_dir=$1
    local formatter_name=$2
    local job_id=$3
    
    echo "[$job_id] Starting: $formatter_name"
    
    # Enhanced directory validation
    validate_directory "$formatter_dir" "formatter directory for $formatter_name" || return 1
    
    # Create job-specific temp directory
    local job_temp="temp_${job_id}"
    mkdir -p "$job_temp" || {
        echo "ERROR at line ${LINENO}: Could not create job temp directory $job_temp" >&2
        return 1
    }
    
    # Change to formatter directory with error handling
    if ! pushd "$formatter_dir" > /dev/null; then
        echo "ERROR at line ${LINENO}: Could not change to directory $formatter_dir" >&2
        return 1
    fi
    
    # Validate formatter script exists and is executable
    case $formatter_dir in
        "format_espn_games")
            local script_name="format_espn_games.sh"
            ;;
        "format_espn_pbp")
            local script_name="format_espn_pbp.sh"
            ;;
        *)
            echo "ERROR at line ${LINENO}: Unknown formatter directory: $formatter_dir" >&2
            popd > /dev/null
            return 1
            ;;
    esac
    
    validate_executable "$script_name" "formatter script for $formatter_name" || {
        popd > /dev/null
        return 1
    }
    
    # Copy local files map
    if [ -f "../temp/local_files.map" ]; then
        mkdir -p temp || {
            echo "ERROR at line ${LINENO}: Could not create temp directory in $formatter_dir" >&2
            popd > /dev/null
            return 1
        }
        
        cp "../temp/local_files.map" "temp/local_files.map" || {
            echo "ERROR at line ${LINENO}: Could not copy local_files.map to $formatter_dir/temp/" >&2
            popd > /dev/null
            return 1
        }
        echo "   [$job_id] Copied local files map"
    else
        echo "   [$job_id] No local files map found (this is normal)"
    fi
    
    # DISABLE S3 UPLOADS IN INDIVIDUAL FORMATTER SCRIPTS
    export SKIP_S3_UPLOAD=true
    
    # Export EP regeneration flag for PBP formatter
    export FORCE_EP_REGENERATE
    
    # Run formatter with detailed error capture
    case $formatter_dir in
        "format_espn_games")
            echo "   [$job_id] Formatting games data (S3 upload deferred to batch)..."
            # Capture the FORMATTER's status, not tee's. `if cmd | tee log` tests
            # the last element of the pipeline, and tee virtually always exits 0,
            # so a formatter that died was reported as a success.
            ./"$script_name" $START_YEAR $END_YEAR 2>&1 | tee "../${job_temp}/games_format.log"
            local formatter_status=${PIPESTATUS[0]}
            if [ "$formatter_status" -eq 0 ]; then
                echo "   SUCCESS [$job_id] Games formatting completed - files ready for batch upload"
            else
                local exit_code=$formatter_status
                echo "ERROR at line ${LINENO}: Games formatter failed with exit code $exit_code" >&2
                echo "Check log: ${job_temp}/games_format.log" >&2
                echo "Last 10 lines of log:" >&2
                tail -n 10 "../${job_temp}/games_format.log" >&2 2>/dev/null || echo "Could not read log file" >&2
                unset SKIP_S3_UPLOAD
                unset FORCE_EP_REGENERATE
                popd > /dev/null
                return 1
            fi
            ;;
            
        "format_espn_pbp")
            echo "   [$job_id] Formatting PBP data with Expected Points Model..."
            echo "   [$job_id] EP Model Regeneration: $FORCE_EP_REGENERATE"
            
            # Run with EP model regeneration flag
            # As above: read PIPESTATUS[0] so tee's exit code cannot mask a failure.
            FORCE_EP_REGENERATE=$FORCE_EP_REGENERATE ./"$script_name" $START_YEAR $END_YEAR 2>&1 | tee "../${job_temp}/pbp_format.log"
            local formatter_status=${PIPESTATUS[0]}
            if [ "$formatter_status" -eq 0 ]; then
                echo "   SUCCESS [$job_id] PBP formatting with EPA completed - files ready for batch upload"
                
                # Check if EPA was actually added
                if [ -f "temp/pbp_edit.csv" ]; then
                    if head -n 1 temp/pbp_edit.csv | grep -q "epa"; then
                        echo "   [$job_id] Confirmed: EPA column present in output"
                    else
                        echo "   [$job_id] Warning: EPA column not found in output"
                    fi
                fi
            else
                local exit_code=$formatter_status
                echo "ERROR at line ${LINENO}: PBP formatter failed with exit code $exit_code" >&2
                echo "Check log: ${job_temp}/pbp_format.log" >&2
                echo "Last 10 lines of log:" >&2
                tail -n 10 "../${job_temp}/pbp_format.log" >&2 2>/dev/null || echo "Could not read log file" >&2
                unset SKIP_S3_UPLOAD
                unset FORCE_EP_REGENERATE
                popd > /dev/null
                return 1
            fi
            ;;
    esac
    
    # Clean up environment variables
    unset SKIP_S3_UPLOAD
    unset FORCE_EP_REGENERATE
    
    # Return to original directory
    if ! popd > /dev/null; then
        echo "ERROR at line ${LINENO}: Could not return to original directory" >&2
        return 1
    fi
    
    echo "FINISHED [$job_id] $formatter_name"
    return 0
}

# ==============================================
# BATCH UPLOAD FOR FORMAT STAGE
# ==============================================
# NOTE: This function is currently disabled but kept for reference
: <<'END_COMMENT'
format_batch_upload() {
    # Check if running from ETL master script
    if [ "$SKIP_S3_UPLOAD" = true ]; then
        echo ""
        echo "=========================================="
        echo "FORMAT S3 UPLOAD DEFERRED TO ETL BATCH"
        echo "=========================================="
        echo "Format stage files saved locally for ETL batch upload"
        
        # Show what's ready
        format_files_ready=0
        if [ -f "format_espn_games/temp/games.csv" ]; then
            games_rows=$(wc -l < format_espn_games/temp/games.csv) || games_rows="unknown"
            echo "  Games formatted: games.csv ($games_rows rows)"
            format_files_ready=$((format_files_ready + 1))
        fi
        
        if [ -f "format_espn_pbp/temp/pbp_edit.csv" ]; then
            pbp_rows=$(wc -l < format_espn_pbp/temp/pbp_edit.csv) || pbp_rows="unknown"
            echo "  PBP formatted: pbp_edit.csv ($pbp_rows rows)"
            format_files_ready=$((format_files_ready + 1))
        fi
        
        echo "  Files ready for ETL batch upload: $format_files_ready"
        echo "ETL batch upload will handle S3 transfer efficiently"
        return 0
    fi
    
    # STANDALONE MODE: Upload format results when format.sh runs alone
    echo ""
    echo "=========================================="
    echo "FORMAT BATCH UPLOAD - ESSENTIAL DATA ONLY"
    echo "=========================================="
    
    local upload_success=true
    local files_uploaded=0
   
    # Upload formatted games file
    if [ -f "format_espn_games/temp/games.csv" ]; then
        games_rows=$(wc -l < format_espn_games/temp/games.csv) || games_rows="unknown"
        echo "1. Uploading formatted games data..."
        echo "   games.csv ($games_rows rows)"
        
        # Local backup first
        mkdir -p ../data/games/formatted 2>/dev/null || true
        cp format_espn_games/temp/games.csv ../data/games/formatted/ 2>/dev/null || true
        
        if aws s3 cp format_espn_games/temp/games.csv s3://ncaaf-data/espn-games-data/games/csvs/games_formatted.csv; then
            echo "   SUCCESS: Games data uploaded to S3"
            files_uploaded=$((files_uploaded + 1))
        else
            echo "   WARNING: Games upload failed"
            upload_success=false
        fi
    fi
    
    # Upload formatted PBP file
    if [ -f "format_espn_pbp/temp/pbp_edit.csv" ]; then
        pbp_rows=$(wc -l < format_espn_pbp/temp/pbp_edit.csv) || pbp_rows="unknown"
        echo ""
        echo "2. Uploading formatted PBP data..."
        echo "   pbp_edit.csv ($pbp_rows rows)"
        
        # Local backup first
        mkdir -p ../data/pbp/formatted 2>/dev/null || true
        cp format_espn_pbp/temp/pbp_edit.csv ../data/pbp/formatted/ 2>/dev/null || true
        
        if aws s3 cp format_espn_pbp/temp/pbp_edit.csv s3://ncaaf-data/espn-pbp-data/pbp/csvs/pbp_formatted.csv; then
            echo "   SUCCESS: PBP data uploaded to S3"
            files_uploaded=$((files_uploaded + 1))
        else
            echo "   WARNING: PBP upload failed"
            upload_success=false
        fi
    fi
   
    echo ""
    echo "=========================================="
    if [ "$upload_success" = true ] && [ $files_uploaded -gt 0 ]; then
        echo "FORMAT BATCH UPLOAD COMPLETE"
        echo "Successfully uploaded $files_uploaded formatted dataset(s)"
    else
        echo "FORMAT BATCH UPLOAD HAD ISSUES"
        echo "Local backups created, but S3 upload had problems"
    fi
    echo "=========================================="
    
    return 0
}
END_COMMENT

# ==============================================
# MAIN EXECUTION
# ==============================================

main() {
    local start_time
    start_time=$(date +%s) || {
        echo "ERROR at line ${LINENO}: Could not get start time" >&2
        return 1
    }
    
    echo "Format Configuration:"
    echo "   Year Range: $START_YEAR to $END_YEAR"
    echo "   Zero-Copy Enabled: $ENABLE_ZERO_COPY"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   EP Model Regeneration: $FORCE_EP_REGENERATE"
    echo ""
    
    # Setup with error handling
    mkdir -p temp || {
        echo "ERROR at line ${LINENO}: Could not create temp directory" >&2
        return 1
    }
    
    rm -f temp/local_files.map || {
        echo "WARNING at line ${LINENO}: Could not remove existing local_files.map (continuing)" >&2
    }
    
    # Discover local files for zero-copy optimization
    if [ "$ENABLE_ZERO_COPY" = true ]; then
        discover_local_files || {
            echo "ERROR at line ${LINENO}: Local file discovery failed" >&2
            return 1
        }
    fi
    
    # Formatters to run
    FORMATTERS=(
        "format_espn_games:ESPN Games Formatting:1"
        "format_espn_pbp:ESPN PBP Formatting with EPA:2"
    )
    
    # Track results
    successful_formatters=()
    failed_formatters=()
    
    # Run formatters (can be parallelized if needed)
    for formatter in "${FORMATTERS[@]}"; do
        IFS=':' read -r dir name job_id <<< "$formatter" || {
            echo "ERROR at line ${LINENO}: Could not parse formatter configuration: $formatter" >&2
            failed_formatters+=("$name (parse error)")
            continue
        }
        
        echo "Processing formatter: $dir -> $name (job $job_id)"
        
        if run_formatter_parallel "$dir" "$name" "$job_id"; then
            successful_formatters+=("$name")
        else
            failed_formatters+=("$name")
        fi
    done
    
    # Format stage batch upload (currently disabled)
    # format_batch_upload || {
    #     echo "WARNING at line ${LINENO}: Batch upload had issues (continuing)" >&2
    # }
    
    # Calculate timing
    local end_time duration minutes seconds
    end_time=$(date +%s) || {
        echo "WARNING at line ${LINENO}: Could not get end time" >&2
        end_time=$start_time
    }
    
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    # Summary
    echo ""
    echo "=================================="
    echo "FORMATTING SUMMARY"
    echo "=================================="
    echo "Total time: ${minutes}m ${seconds}s"
    echo "EP Model: ${FORCE_EP_REGENERATE}"
    
    # Check for EP model artifacts
    if [ -f "format_espn_pbp/expected_points_lookup_table.csv" ]; then
        ep_rows=$(wc -l < format_espn_pbp/expected_points_lookup_table.csv)
        echo "EP Lookup Table: $((ep_rows - 1)) scenarios"
    fi
    
    if [ ${#successful_formatters[@]} -gt 0 ]; then
        echo ""
        echo "SUCCESS: Successful formatters (${#successful_formatters[@]}):"
        for formatter in "${successful_formatters[@]}"; do
            echo "   - $formatter"
        done
    fi
    
    if [ ${#failed_formatters[@]} -gt 0 ]; then
        echo ""
        echo "FAILED: Failed formatters (${#failed_formatters[@]}):"
        for formatter in "${failed_formatters[@]}"; do
            echo "   - $formatter"
        done
    fi
    
    # Show error summary if any occurred
    if [ $TOTAL_ERRORS -gt 0 ]; then
        echo ""
        echo "ERROR SUMMARY ($TOTAL_ERRORS total errors):"
        echo "=================================="
        for error in "${ERROR_LOG[@]}"; do
            echo "   $error"
        done
    fi
    
    # Cleanup
    rm -rf temp_* 2>/dev/null || {
        echo "WARNING at line ${LINENO}: Could not clean up temporary directories" >&2
    }
    
    if [ ${#successful_formatters[@]} -gt 0 ]; then
        echo ""
        echo "SUCCESS: Data formatting with Expected Points completed successfully!"
        return 0
    else
        echo ""
        echo "ERROR: All formatters failed"
        return 1
    fi
}

# Run main function with argument passing
main "$@"