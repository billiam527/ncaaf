import os
import json
import pandas as pd
import datetime
import argparse
import logging
import sys
import time
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import numpy as np
from tqdm.asyncio import tqdm
from date_generation import date_list_generation
from json_to_csv import transform_espn_ncaaf_data

# Largest slice of games handed to a single worker process.
MAX_CHUNK_GAMES = 100

# Rough peak resident size per worker while transforming a chunk of this size.
# Measured at ~2.2 GB RSS per worker during the 2016 season; kept conservative
# because under-estimating costs a killed worker and a silently short season.
_WORKER_MEM_BYTES = 2_200_000_000


def _memory_safe_worker_count():
    """How many workers currently-available RAM can support.

    Reads MemAvailable rather than total memory so a busy machine scales down
    instead of overcommitting. Falls back to a conservative 2 if unreadable.
    """
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) * 1024
                    # Leave roughly a quarter of available RAM as headroom for
                    # the parent process, which holds the full JSON list.
                    usable = available * 0.75
                    return max(1, int(usable // _WORKER_MEM_BYTES))
    except Exception:
        pass
    return 2

class OptimizedESPNCollector:
    def __init__(self, max_concurrent=50, chunk_size=1000, timeout=10):
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.session = None
        self.semaphore = None
        
    async def __aenter__(self):
        # Optimized connection settings
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout, connect=5)
        
        # No User-Agent override: aiohttp's default is accepted by ESPN, while a
        # spoofed browser UA gets a blanket 403.
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_with_retry(self, url, max_retries=2):
        """Fetch URL with minimal retry logic"""
        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 404:
                            return None  # No data available
                        else:
                            if attempt == max_retries:
                                return None
                            await asyncio.sleep(0.1 * (attempt + 1))
                except asyncio.TimeoutError:
                    if attempt == max_retries:
                        return None
                    await asyncio.sleep(0.1 * (attempt + 1))
                except Exception:
                    if attempt == max_retries:
                        return None
                    await asyncio.sleep(0.1 * (attempt + 1))
        return None
    
    async def collect_game_ids_batch(self, date_urls):
        """Collect game IDs from date URLs using async batch processing"""
        print("Collecting game IDs...")
        
        async def process_date_url(url):
            data = await self.fetch_with_retry(url)
            if data and 'events' in data:
                return [event.get('id') for event in data.get('events', []) if event.get('id')]
            return []
        
        # Process in chunks with progress bar
        all_game_ids = []
        
        with tqdm(total=len(date_urls), desc="Fetching game IDs") as pbar:
            for i in range(0, len(date_urls), self.chunk_size):
                chunk = date_urls[i:i + self.chunk_size]
                
                # Process chunk concurrently
                tasks = [process_date_url(url) for url in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in results:
                    if isinstance(result, list):
                        all_game_ids.extend(result)
                
                pbar.update(len(chunk))
                pbar.set_postfix(games_found=len(all_game_ids))
                
                # Brief pause between chunks to be respectful
                if i + self.chunk_size < len(date_urls):
                    await asyncio.sleep(0.05)
        
        return all_game_ids
    
    async def collect_pbp_data_batch(self, game_urls):
        """Collect play-by-play data using async batch processing"""
        print(f"Collecting play-by-play data for {len(game_urls)} games...")
        
        async def process_game_url(url):
            try:
                game_id = url.split('event=')[1].split('&')[0]
                
                # Fetch data
                data = await self.fetch_with_retry(url)
                if not data:
                    return None
                
                # Save JSON asynchronously
                json_path = f'temp/pbpjsons/json_pbp_{game_id}.json'
                async with aiofiles.open(json_path, 'w') as f:
                    await f.write(json.dumps(data))
                
                return data
                
            except Exception:
                return None
        
        # Collect JSON data
        all_json_data = []
        
        with tqdm(total=len(game_urls), desc="Fetching PBP data") as pbar:
            for i in range(0, len(game_urls), self.chunk_size):
                chunk = game_urls[i:i + self.chunk_size]
                
                # Process chunk concurrently
                tasks = [process_game_url(url) for url in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect non-None results
                valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
                all_json_data.extend(valid_results)
                
                pbar.update(len(chunk))
                pbar.set_postfix(games_processed=len(all_json_data))
                
                # Brief pause between chunks
                if i + self.chunk_size < len(game_urls):
                    await asyncio.sleep(0.05)
        
        return all_json_data

def process_json_data_parallel(json_data_list, max_workers=None):
    """Process JSON data to DataFrames using multiprocessing"""
    if not json_data_list:
        return []
    
    print(f"Converting {len(json_data_list)} games to DataFrames...")
    
    # Memory, not CPU, is the binding constraint here: each worker receives a
    # pickled copy of its chunk while the parent still holds the full list, so
    # peak usage is roughly (1 + max_workers) x chunk size. Running one worker
    # per core previously exhausted RAM, the kernel killed workers mid-flight,
    # and whole chunks of a season were silently dropped.
    mem_workers = _memory_safe_worker_count()
    if max_workers is None:
        max_workers = min(len(json_data_list), os.cpu_count() or 4, mem_workers)
    else:
        max_workers = min(max_workers, mem_workers)
    max_workers = max(1, max_workers)

    # Cap chunk size independently of worker count so a season with many games
    # does not hand each worker an unbounded slice.
    chunk_size = max(1, min(MAX_CHUNK_GAMES, len(json_data_list) // max_workers or 1))
    chunks = [json_data_list[i:i + chunk_size] for i in range(0, len(json_data_list), chunk_size)]

    print(f"  using {max_workers} worker(s), {len(chunks)} chunk(s) of <= {chunk_size} games")

    all_dfs = []
    failed_games = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_json_chunk, chunk): i for i, chunk in enumerate(chunks)}

        with tqdm(total=len(json_data_list), desc="Processing JSON") as pbar:
            for future in future_to_chunk:
                idx = future_to_chunk[future]
                try:
                    chunk_dfs = future.result()
                    all_dfs.extend(chunk_dfs)
                except BrokenProcessPool as e:
                    # A worker was killed (almost always OOM). Retry this chunk
                    # in-process rather than dropping it - silently returning a
                    # truncated season is far worse than being slow.
                    logging.warning(f"Worker died on chunk {idx} ({e}); retrying serially")
                    try:
                        all_dfs.extend(process_json_chunk(chunks[idx]))
                    except Exception as retry_err:
                        failed_games += len(chunks[idx])
                        logging.error(f"Serial retry of chunk {idx} also failed: {retry_err}")
                except Exception as e:
                    failed_games += len(chunks[idx])
                    logging.error(f"Error processing chunk {idx}: {e}")
                finally:
                    pbar.update(len(chunks[idx]))

    if failed_games:
        # Refuse to pass off a partial season as a complete one.
        raise RuntimeError(
            f"JSON processing lost {failed_games} of {len(json_data_list)} games; "
            f"refusing to write a truncated result"
        )

    return all_dfs

def process_json_chunk(json_chunk):
    """Process a chunk of JSON data - for multiprocessing"""
    dfs = []
    for json_data in json_chunk:
        try:
            df = transform_espn_ncaaf_data(json_data)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception:
            continue
    return dfs

def setup_logging_optimized():
    """Setup minimal logging for performance"""
    os.makedirs('temp', exist_ok=True)
    
    logging.basicConfig(
        level=logging.WARNING,  # Only warnings and errors
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('temp/pbp_collection.log', mode='w'),
            logging.StreamHandler(sys.stderr)
        ]
    )

def create_urls_fast(prefix, suffix, data_list):
    """Fast URL creation using list comprehension"""
    return [f"{prefix}{item}{suffix}" for item in data_list]

def combine_dataframes_optimized(dataframes):
    """Optimized DataFrame combination"""
    if not dataframes:
        return pd.DataFrame()
    
    print("Combining DataFrames...")
    
    # Use concat with optimized parameters
    combined = pd.concat(
        dataframes,
        ignore_index=True,
        copy=False,  # Don't copy data unnecessarily
        sort=False   # Don't sort columns
    )
    
    # Optimize data types to reduce memory
    for col in combined.columns:
        if combined[col].dtype == 'object':
            try:
                # Try to convert to numeric if possible
                numeric_col = pd.to_numeric(combined[col], errors='ignore')
                if numeric_col.dtype != 'object':
                    combined[col] = numeric_col
            except:
                pass
    
    return combined

async def main_async(start_date, end_date):
    """Main async function for optimized collection"""
    setup_logging_optimized()
    
    print(f"\nOPTIMIZED ESPN Collection")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Setup
        os.makedirs('temp', exist_ok=True)
        os.makedirs('temp/pbpjsons', exist_ok=True)
        
        # Step 1: Generate dates (fast)
        print("Step 1: Generating dates...")
        dates = date_list_generation(start_date, end_date)
        print(f"   SUCCESS: Generated {len(dates)} dates")
        
        # Step 2: Create date URLs (fast)
        print("Step 2: Creating date URLs...")
        date_urls = create_urls_fast(
            'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=',
            '&limit=100',
            dates
        )
        print(f"   SUCCESS: Created {len(date_urls)} URLs")
        
        # Step 3: Async game ID collection
        async with OptimizedESPNCollector(max_concurrent=50, timeout=8) as collector:
            game_ids = await collector.collect_game_ids_batch(date_urls)
        
        if not game_ids:
            print("WARNING: No games found in date range")
            # Create empty file
            empty_df = pd.DataFrame(columns=['game_id', 'play_text'])
            empty_df.to_csv(f'temp/play-by-play_{start_date}_to_{end_date}.csv')
            return
        
        print(f"   SUCCESS: Found {len(game_ids)} games")
        
        # Step 4: Create game URLs (fast)
        print("Step 3: Creating game URLs...")
        game_urls = create_urls_fast(
            'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event=',
            '&limit=100',
            game_ids
        )
        print(f"   SUCCESS: Created {len(game_urls)} game URLs")
        
        # Step 5: Async PBP data collection
        async with OptimizedESPNCollector(max_concurrent=40, timeout=10) as collector:
            json_data_list = await collector.collect_pbp_data_batch(game_urls)
        
        if not json_data_list:
            print("WARNING: No play-by-play data found")
            empty_df = pd.DataFrame(columns=['game_id', 'play_text'])
            empty_df.to_csv(f'temp/play-by-play_{start_date}_to_{end_date}.csv')
            return
        
        print(f"   SUCCESS: Collected data for {len(json_data_list)} games")
        
        # Step 6: Parallel JSON processing. Worker count is decided inside based
        # on available memory - passing os.cpu_count() here previously OOMed.
        dataframes = process_json_data_parallel(json_data_list)
        
        if not dataframes:
            print("WARNING: No valid play data processed")
            empty_df = pd.DataFrame(columns=['game_id', 'play_text'])
            empty_df.to_csv(f'temp/play-by-play_{start_date}_to_{end_date}.csv')
            return
        
        # Step 7: Optimized DataFrame combination and save
        final_df = combine_dataframes_optimized(dataframes)
        
        if not final_df.empty:
            # Set index if possible
            if 'id' in final_df.columns:
                final_df.set_index('id', inplace=True)
            
            # Save with optimized settings
            output_file = f'temp/play-by-play_{start_date}_to_{end_date}.csv'
            print("Step 4: Saving results...")
            
            # Use optimized CSV writing
            final_df.to_csv(
                output_file,
                chunksize=50000,  # Large chunks for efficiency
                compression=None   # No compression for speed
            )
            
            # Results
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\nSUCCESS: Collection Complete!")
            print(f"Total plays: {len(final_df):,}")
            print(f"Games processed: {final_df['game_id'].nunique() if 'game_id' in final_df.columns else len(json_data_list)}")
            print(f"Time: {duration:.1f} seconds")
            print(f"Speed: {len(final_df) / duration:.0f} plays/second")
            print(f"File: {output_file}")
            
        else:
            print("WARNING: No data to save")
            
    except Exception as e:
        print(f"ERROR: {e}")
        raise

def main(start_date, end_date):
    """Wrapper to run async main"""
    return asyncio.run(main_async(start_date, end_date))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized ESPN Play-by-Play Collector')
    
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    default_start = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    
    parser.add_argument('--start_date', default=default_start, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=default_start, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        main(args.start_date, args.end_date)
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nCollection failed: {e}")
        sys.exit(1)