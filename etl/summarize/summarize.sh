#!/usr/bin/env bash
set -e

# ==============================================
# CONFIGURATION
# ==============================================

STATS="play_success rush_success pass_success yards_per_play rush_yards_per_play pass_yards_per_play explosive_play_rate explosive_rush_rate explosive_pass_rate epa_per_play epa_per_rush epa_per_pass"
ENABLE_SMART_CACHING=true
INCREMENTAL_MODE=true
MAX_PARALLEL_JOBS=3
RUN_DERIVED=true
DERIVED_ONLY=false

# Where to look for games/pbp/teams inputs, in priority order.
# Fresh format/ output wins; ../data/*/formatted is the on-disk fallback holding
# the last good formatted data when collect/format did not run this cycle.
SEARCH_PATHS=(
    "../format/format_espn_games/temp"
    "../format/format_espn_pbp/temp"
    "../../format/format_espn_games/temp"
    "../../format/format_espn_pbp/temp"
    "../data/games/formatted"
    "../data/pbp/formatted"
    "../../data/games/formatted"
    "../../data/pbp/formatted"
    "../collect/collect_espn_teams/temp"
    "../../collect/collect_espn_teams/temp"
    "./temp"  # Previous runs
)

# The ../data/*/formatted paths are a last-resort snapshot, not pipeline output.
# Falling back to them silently is how a dead formatter gets mistaken for a
# healthy run, so say so loudly and include the file's age.
warn_if_stale_fallback() {
    local path="$1" file="$2"
    case "$path" in
        */data/*/formatted)
            local age_days
            age_days=$(( ( $(date +%s) - $(stat -c %Y "$path/$file") ) / 86400 ))
            echo "" >&2
            echo "  ***********************************************************" >&2
            echo "  WARNING: using the STATIC SNAPSHOT for $file" >&2
            echo "           $path/$file (${age_days} days old)" >&2
            echo "           The format stage produced no output for this file." >&2
            echo "           Results will reflect stale data - fix the formatter." >&2
            echo "  ***********************************************************" >&2
            echo "" >&2
            ;;
    esac
}

# Print where we looked, so an acquisition failure diagnoses itself.
report_search_paths() {
    local what="$1"
    echo "  Could not locate $what locally or in S3."
    echo "  Searched these paths (relative to $(pwd)):"
    local p
    for p in "${SEARCH_PATHS[@]}"; do
        if [ -d "$p" ]; then
            echo "    [dir exists] $p"
        else
            echo "    [missing]    $p"
        fi
    done
    echo "  S3 fallback also failed - the object may not exist in the bucket."
    echo "  Fix: run the collect/format stages, or place the file in ../data/*/formatted/."
}

# Setup logging
LOGFILE="logs/summarize_bash.log"
mkdir -p logs
exec > >(tee -a "$LOGFILE")
exec 2>&1

echo "$(date): Starting optimized summarize script"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            INCREMENTAL_MODE=false
            echo "Full mode enabled - will reprocess all data"
            shift
            ;;
        --no-cache)
            ENABLE_SMART_CACHING=false
            echo "Cache disabled - will re-download all files"
            shift
            ;;
        --parallel-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --skip-derived)
            RUN_DERIVED=false
            echo "Derived summaries disabled - season_summaries only"
            shift
            ;;
        --derived-only)
            DERIVED_ONLY=true
            echo "Derived summaries only - season_summaries will not be rebuilt"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full] [--no-cache] [--parallel-jobs N]"
            echo "          [--skip-derived] [--derived-only]"
            exit 1
            ;;
    esac
done

# Setup temp directory
rm -rf temp 2>/dev/null || true
mkdir temp
mkdir -p results

echo "Configuration: Incremental=$INCREMENTAL_MODE, Cache=$ENABLE_SMART_CACHING, Jobs=$MAX_PARALLEL_JOBS"

# ==============================================
# SMART LOCAL FILE DISCOVERY
# ==============================================

discover_local_data() {
    echo "Discovering local data files..."
    
    # Track which files we successfully found and linked
    declare -A files_found
    files_found["games"]=false
    files_found["pbp"]=false  
    files_found["teams"]=false
    
    local_files_found=0
    
    local search_paths=("${SEARCH_PATHS[@]}")

    # Games file discovery
    for path in "${search_paths[@]}"; do
        if [ -f "$path/games.csv" ] && [ -s "$path/games.csv" ]; then
            echo "Found games file: $path/games.csv" >&2
            warn_if_stale_fallback "$path" "games.csv"
            abs_path=$(readlink -f "$path/games.csv")
            ln -sf "$abs_path" "temp/games.csv"
            files_found["games"]=true
            local_files_found=$((local_files_found + 1))
            break
        fi
    done
    
    # PBP file discovery
    for path in "${search_paths[@]}"; do
        if [ -f "$path/pbp_edit.csv" ] && [ -s "$path/pbp_edit.csv" ]; then
            echo "Found PBP file: $path/pbp_edit.csv" >&2
            warn_if_stale_fallback "$path" "pbp_edit.csv"
            abs_path=$(readlink -f "$path/pbp_edit.csv")
            ln -sf "$abs_path" "temp/pbp.csv"
            files_found["pbp"]=true
            local_files_found=$((local_files_found + 1))
            break
        elif [ -f "$path/pbp.csv" ] && [ -s "$path/pbp.csv" ]; then
            echo "Found PBP file: $path/pbp.csv" >&2
            abs_path=$(readlink -f "$path/pbp.csv")
            ln -sf "$abs_path" "temp/pbp.csv"
            files_found["pbp"]=true
            local_files_found=$((local_files_found + 1))
            break
        fi
    done
    
    # Teams file discovery
    for path in "${search_paths[@]}"; do
        if [ -f "$path/teams.csv" ] && [ -s "$path/teams.csv" ]; then
            echo "Found teams file: $path/teams.csv" >&2
            abs_path=$(readlink -f "$path/teams.csv")
            ln -sf "$abs_path" "temp/teams.csv"
            files_found["teams"]=true
            local_files_found=$((local_files_found + 1))
            break
        fi
    done
    
    echo "Local discovery complete: $local_files_found files found" >&2
    
    # Export the findings for download_missing_files to use
    echo "${files_found["games"]}" > temp/.games_found
    echo "${files_found["pbp"]}" > temp/.pbp_found  
    echo "${files_found["teams"]}" > temp/.teams_found
    
    return $local_files_found
}

# ==============================================
# DOWNLOAD FUNCTION
# ==============================================

download_missing_files() {
    echo "Checking for missing files..."
    
    # Read what files were found locally
    games_found=false
    pbp_found=false
    teams_found=false
    
    if [ -f "temp/.games_found" ]; then
        games_found=$(cat temp/.games_found)
    fi
    if [ -f "temp/.pbp_found" ]; then
        pbp_found=$(cat temp/.pbp_found)
    fi
    if [ -f "temp/.teams_found" ]; then
        teams_found=$(cat temp/.teams_found)
    fi
    
    # Check what files need downloading
    download_jobs=()
    
    if [ "$games_found" = false ]; then
        download_jobs+=("games")
    fi
    
    if [ "$pbp_found" = false ]; then
        download_jobs+=("pbp") 
    fi
    
    if [ "$teams_found" = false ]; then
        download_jobs+=("teams")
    fi
    
    if [ ${#download_jobs[@]} -eq 0 ]; then
        echo "All files available locally"
        return 0
    fi
    
    echo "Downloading ${#download_jobs[@]} missing files..."
    
    # Download missing files
    for job in "${download_jobs[@]}"; do
        case $job in
            "games")
                echo "Downloading games.csv..." >&2
                if aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/games.csv temp/games.csv --quiet; then
                    echo "Downloaded games.csv" >&2
                else
                    echo "ERROR: Failed to download games.csv" >&2
                    report_search_paths "games.csv" >&2
                    return 1
                fi
                ;;
            "pbp")
                echo "Downloading pbp_edit.csv..." >&2
                if aws s3 cp s3://ncaaf-data/espn-pbp-data/pbp/csvs/pbp_edit.csv temp/pbp.csv --quiet; then
                    echo "Downloaded pbp.csv" >&2
                else
                    echo "ERROR: Failed to download pbp.csv" >&2
                    report_search_paths "pbp_edit.csv / pbp.csv" >&2
                    return 1
                fi
                ;;
            "teams")
                echo "Downloading teams.csv..." >&2
                if aws s3 cp s3://ncaaf-data/espn-teams-data/teams.csv temp/teams.csv --quiet; then
                    echo "Downloaded teams.csv" >&2
                else
                    echo "WARNING: teams.csv not found - creating minimal file" >&2
                    echo "id,name" > temp/teams.csv
                fi
                ;;
        esac
    done
    
    return 0
}

# ==============================================
# DATA ACQUISITION
# ==============================================

acquire_data() {
    echo "Starting data acquisition..."
    
    # Try local files first if caching enabled
    local_files_count=0
    if [ "$ENABLE_SMART_CACHING" = true ]; then
        discover_local_data
        local_files_count=$?
    fi
    
    # Download any missing files
    if ! download_missing_files; then
        echo "ERROR: Failed to download required files"
        return 1
    fi
    
    # Verify all required files exist and are readable
    required_files=("temp/games.csv" "temp/pbp.csv")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ] || [ ! -s "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "ERROR: Missing or empty required files: ${missing_files[*]}"
        return 1
    fi
    
    # Show file sizes for verification
    echo "Data files ready:"
    for file in temp/*.csv; do
        if [ -f "$file" ]; then
            rows=$(wc -l < "$file" 2>/dev/null || echo "0")
            echo "  $(basename $file): $rows rows"
        fi
    done
    
    echo "Data acquisition complete"
    return 0
}

# ==============================================
# INCREMENTAL PROCESSING CHECK
# ==============================================

check_incremental_processing() {
    if [ "$INCREMENTAL_MODE" = false ]; then
        echo "Full mode - processing all data"
        return 1  # Process everything
    fi
    
    echo "Checking for incremental processing opportunities..."
    
    # Check if output files exist and are recent
    local files_to_check=(
        "results/game_by_game_summaries.csv"
        "results/season_summaries.csv" 
        "results/rolling_summaries.csv"
    )
    
    local all_exist=true
    
    for file in "${files_to_check[@]}"; do
        if [ ! -f "$file" ]; then
            all_exist=false
            echo "Missing output file: $(basename $file)"
            break
        fi
    done
    
    if [ "$all_exist" = false ]; then
        echo "Output files missing - full processing required"
        return 1
    fi
    
    # Check if input data is newer than output
    local input_files=("temp/games.csv" "temp/pbp.csv")
    local newest_input=""
    local newest_output=""
    
    # Find newest input file
    for file in "${input_files[@]}"; do
        if [ -f "$file" ] && [[ "$file" -nt "$newest_input" ]]; then
            newest_input="$file"
        fi
    done
    
    # Find newest output file
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ] && [[ "$file" -nt "$newest_output" ]]; then
            newest_output="$file"
        fi
    done
    
    if [[ "$newest_input" -nt "$newest_output" ]]; then
        echo "Input data newer than output - processing required"
        return 1
    fi
    
    echo "Output files are up-to-date - skipping processing"
    return 0
}

# ==============================================
# ANALYSIS EXECUTION
# ==============================================

run_analysis() {
    echo "Starting analysis..."
    
    # Check file sizes for memory optimization
    games_size=$(wc -l < temp/games.csv)
    pbp_size=$(wc -l < temp/pbp.csv)
    
    echo "Data sizes: Games=$games_size, PBP=$pbp_size"
    
    # Build analysis command
    local analysis_cmd="python summarize_games.py"
    analysis_cmd+=" --pbp_file temp/pbp.csv"
    analysis_cmd+=" --games_file temp/games.csv"
    analysis_cmd+=" --teams_file temp/teams.csv"
    analysis_cmd+=" --statistics $STATS"
    analysis_cmd+=" --output_dir results"
    # Garbage-time plays are kept. Removing them was worth -0.087 MAE against
    # keeping them (t=-4.14) on a 2019-2025 walk-forward - discarding 11% of
    # plays makes the opponent-adjusted estimates noisier by more than the
    # blowout snaps mislead.
    analysis_cmd+=" --include_garbage_time"
    analysis_cmd+=" --alpha 1.0"
    
    echo "Running analysis pipeline..."
    
    # Execute with timing
    start_time=$(date +%s)
    
    if eval $analysis_cmd; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$((duration / 60))
        seconds=$((duration % 60))
        
        echo "Analysis completed in ${minutes}m ${seconds}s"
        return 0
    else
        echo "ERROR: Analysis failed"
        return 1
    fi
}

# ==============================================
# DERIVED SUMMARIES
# ==============================================
#
# summarize_games.py produces season_summaries and the rolling files. Everything
# else in results/ is derived from those plus the play-by-play, and until this
# stage existed every one of them was run by hand. That left the directory
# holding files built five days apart - season_summaries from one afternoon,
# ol_production from three days later off the same play-by-play, returning
# production older than both - with nothing to say so.
#
# It matters now because the preseason model consumes returning_production,
# team_talent and position_ratings directly. A missing position_ratings.csv does
# not fail loudly at this end: the merge in preprocess.py silently returns the
# frame unchanged, the feature matrix comes out narrower than the scaler, and the
# error surfaces at prediction time as a width mismatch.
#
# Order is dependency order, not preference. Anything in a later stage reads a
# file an earlier stage writes.
#
#   1  independent scans of the play-by-play
#   2  needs stage 1, or reads the CFBD player files directly
#   3  needs season_summaries plus a stage-2 file
#   4  needs front_seven
#   5  needs every projection above it
#
# A stage-1 or stage-2 failure is reported and the run continues, because these
# are independent of one another and a missing CFBD player file should not cost
# the whole rebuild. position_ratings is different: if it fails the model cannot
# be trained, so the function returns non-zero.

# special_teams reads only the play-by-play and games, so it belongs in the
# first stage alongside the other producers. position_ratings consumes it.
DERIVED_STAGE_1="havoc qb_production receiver_production rb_production
                 ol_production drive_factors run_pass_ratio special_teams"
DERIVED_STAGE_2="adjust_havoc talent_by_position returning_production
                 team_talent roster_talent qb_projection receiver_projection
                 rb_projection"
DERIVED_STAGE_3="ol_projection front_seven"
DERIVED_STAGE_4="defensive_backs"
DERIVED_STAGE_5="position_ratings"

DERIVED_FAILED=""
DERIVED_OK=""

run_derived_module() {
    local mod="$1"
    if [ ! -f "${mod}.py" ]; then
        echo "    ${mod}: no such module, skipped"
        return 0
    fi
    local t0 t1
    t0=$(date +%s)
    if python "${mod}.py" > "temp/${mod}.log" 2>&1; then
        t1=$(date +%s)
        echo "    ${mod}: ok ($((t1 - t0))s)"
        DERIVED_OK+=" ${mod}"
        return 0
    fi
    t1=$(date +%s)
    echo "    ${mod}: FAILED after $((t1 - t0))s - temp/${mod}.log"
    tail -3 "temp/${mod}.log" | sed 's/^/      /'
    DERIVED_FAILED+=" ${mod}"
    return 1
}

run_derived() {
    echo ""
    echo "Building derived summaries..."
    local start
    start=$(date +%s)

    # Decode the play-by-play once. Eight read sites across seven modules used
    # to tokenise the same 1.3GB CSV independently, at about 45 seconds each;
    # they now read a 0.53GB pickle of the 20 columns they collectively want, in
    # about 5. Every module falls back to the CSV if this is missing or stale,
    # so a failure here costs speed rather than correctness.
    echo "  stage 0: shared play-by-play cache"
    if python pbp_cache.py --pbp temp/pbp.csv > temp/pbp_cache.log 2>&1; then
        sed 's/^/    /' temp/pbp_cache.log
    else
        echo "    cache build FAILED - modules will read the CSV directly"
        tail -3 temp/pbp_cache.log | sed 's/^/      /'
    fi

    local stage n=1
    for stage in "$DERIVED_STAGE_1" "$DERIVED_STAGE_2" "$DERIVED_STAGE_3" \
                 "$DERIVED_STAGE_4" "$DERIVED_STAGE_5"; do
        echo "  stage ${n}:"
        local mod
        for mod in $stage; do
            # set -e is on; these are allowed to fail without killing the run
            run_derived_module "$mod" || true
        done
        n=$((n + 1))
    done

    local end
    end=$(date +%s)
    echo "  derived summaries took $(( (end - start) / 60 ))m $(( (end - start) % 60 ))s"

    if [ -n "$DERIVED_FAILED" ]; then
        echo "  FAILED:$DERIVED_FAILED"
    fi

    # The model cannot be trained without these three, so their absence is the
    # difference between a degraded run and a broken one.
    local required="results/position_ratings.csv results/returning_production.csv
                    results/team_talent.csv"
    local missing="" f
    for f in $required; do
        [ -f "$f" ] || missing+=" $(basename "$f")"
    done
    if [ -n "$missing" ]; then
        echo "  ERROR: the preseason model requires:$missing"
        return 1
    fi
    return 0
}

# ==============================================
# CONDITIONAL RESULTS UPLOAD
# ==============================================

upload_results() {
    # Check if running from ETL master script
    if [ "$SKIP_S3_UPLOAD" = true ]; then
        echo ""
        echo "=========================================="
        echo "SUMMARIZE S3 UPLOAD DEFERRED TO ETL BATCH"
        echo "=========================================="
        echo "Summarization results saved locally for ETL batch upload"
        
        # Show what's ready
        if [ -d "results" ]; then
            result_files=$(find results -name "*.csv" | wc -l)
            echo "  Analysis results ready: $result_files CSV files"
            echo "  Location: $(pwd)/results/"
            
            # Create local backup
            mkdir -p ../data/analysis/backups 2>/dev/null || true
            cp results/*.csv ../data/analysis/backups/ 2>/dev/null || true
            echo "  Local backup: ../data/analysis/backups/"
        fi
        
        echo "ETL batch upload will handle S3 transfer efficiently"
        return 0
    fi
    
    # STANDALONE MODE: Upload results when summarize.sh runs alone
    echo ""
    echo "=========================================="
    echo "SUMMARIZE BATCH UPLOAD - ANALYSIS RESULTS"
    echo "=========================================="
    
    # Check what files were created
    output_files=(
        "results/game_by_game_summaries.csv"
        "results/season_summaries.csv"
        "results/rolling_summaries.csv"
    )
    
    upload_success=true
    files_uploaded=0
    
    for local_file in "${output_files[@]}"; do
        if [ -f "$local_file" ]; then
            file_size=$(wc -l < "$local_file")
            filename=$(basename "$local_file")
            s3_path="s3://ncaaf-data/model_data/$filename"
            
            echo "Uploading $filename ($file_size rows)..."
            
            # Local backup first
            mkdir -p ../data/analysis/results 2>/dev/null || true
            cp "$local_file" ../data/analysis/results/ 2>/dev/null || true
            
            if aws s3 cp "$local_file" "$s3_path" --quiet; then
                echo "   SUCCESS: Uploaded $filename"
                files_uploaded=$((files_uploaded + 1))
            else
                echo "   ERROR: Failed to upload $filename"
                upload_success=false
            fi
        else
            echo "WARNING: Missing output file: $(basename $local_file)"
            upload_success=false
        fi
    done
    
    echo ""
    echo "=========================================="
    if [ "$upload_success" = true ] && [ $files_uploaded -gt 0 ]; then
        echo "SUMMARIZE BATCH UPLOAD COMPLETE"
        echo "Successfully uploaded $files_uploaded analysis result(s)"
    else
        echo "SUMMARIZE BATCH UPLOAD HAD ISSUES"
        echo "Local backups created, but S3 upload had problems"
    fi
    echo "=========================================="
    
    return 0
}

# ==============================================
# MAIN EXECUTION
# ==============================================

main() {
    start_time=$(date +%s)
    
    echo "Optimized College Football Data Summarization"
    echo "Started at: $(date)"
    echo ""
    
    # Step 1: Acquire data (with smart caching)
    if ! acquire_data; then
        echo "ERROR: Data acquisition failed"
        exit 1
    fi

    if [ "$DERIVED_ONLY" = true ]; then
        # Rebuild everything downstream of season_summaries without redoing the
        # opponent-adjustment pass. Note this is not cheap: six stage-1 modules
        # scan the play-by-play independently, so the saving is one pass out of
        # seven, not the bulk of the run.
        if ! run_derived; then
            echo "ERROR: Derived summaries failed"
            exit 1
        fi
        echo ""
        echo "Derived summaries rebuilt."
        exit 0
    fi

    # Step 2: Check for incremental processing
    if check_incremental_processing; then
        echo "Incremental check passed - results are up-to-date"
        echo "Summarization completed (no processing needed)!"
        exit 0
    fi

    # Step 3: Run analysis
    if ! run_analysis; then
        echo "ERROR: Analysis failed"
        exit 1
    fi

    # Step 4: Everything derived from it. Skipping this leaves results/ holding
    # a fresh season_summaries beside stale derived files that were built from
    # an older one, which is harder to notice than an outright failure.
    if [ "$RUN_DERIVED" = true ]; then
        if ! run_derived; then
            echo "ERROR: Derived summaries failed"
            exit 1
        fi
    else
        echo ""
        echo "WARNING: --skip-derived. results/ now holds a fresh"
        echo "         season_summaries beside derived files built from an"
        echo "         older one. The preseason model reads both."
    fi

    # Step 5: Conditional upload results
    if ! upload_results; then
        echo "ERROR: Upload failed"
        exit 1
    fi
    
    # Calculate timing
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    # Summary
    echo ""
    echo "SUMMARIZATION SUMMARY"
    echo "===================="
    echo "Total time: ${minutes}m ${seconds}s"
    echo "Smart caching: $ENABLE_SMART_CACHING"
    echo "Incremental mode: $INCREMENTAL_MODE"
    
    # Show output file stats
    if [ -d "results" ]; then
        echo ""
        echo "Output files created:"
        for file in results/*.csv; do
            if [ -f "$file" ]; then
                rows=$(wc -l < "$file")
                echo "  $(basename $file): $rows rows"
            fi
        done
    fi
    
    echo ""
    echo "Optimized summarization completed successfully!"
    
    # Cleanup tracking files
    rm -f temp/.games_found temp/.pbp_found temp/.teams_found
}

# Run main function
main "$@"