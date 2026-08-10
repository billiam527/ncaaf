#!/usr/bin/env python3
"""
Expected Points Anomaly Detection and Smoothing Script

This script identifies anomalies in expected points lookup tables and applies
smoothing to fix yards_to_goal violations where EP increases as distance from goal increases.

Usage:
    python ep_anomaly_detector.py --input_file expected_points_lookup_table.csv --output_file flagged_expected_points.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EPAnomalyDetector:
    """Class to detect, flag, and smooth anomalies in expected points data"""
    
    def __init__(self, ep_threshold: float = 0.2):
        self.ep_threshold = ep_threshold
        
    def flag_yards_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag situations where expected points increase as yards_to_goal increases"""
        logging.info("Flagging yards_to_goal anomalies...")
        
        # Initialize anomaly flags
        df['yards_anomaly'] = False
        df['yards_anomaly_severity'] = 'none'
        df['yards_anomaly_ep_change'] = 0.0
        
        # Group by time/down/distance and check yards_to_goal trend
        groups = df.groupby(['half_seconds_remaining', 'down', 'distance'])
        
        anomaly_count = 0
        with tqdm(total=len(groups), desc="Checking yards trends", unit="group") as pbar:
            for (time, down, dist), group in groups:
                if len(group) < 2:
                    pbar.update(1)
                    continue
                
                # Sort by yards_to_goal
                group_sorted = group.sort_values('yards_to_goal')
                indices = group_sorted.index
                
                # Check each consecutive pair
                for i in range(1, len(group_sorted)):
                    curr_idx = indices[i]
                    prev_idx = indices[i-1]
                    
                    curr_yards = group_sorted.iloc[i]['yards_to_goal']
                    prev_yards = group_sorted.iloc[i-1]['yards_to_goal']
                    curr_ep = group_sorted.iloc[i]['expected_points']
                    prev_ep = group_sorted.iloc[i-1]['expected_points']
                    
                    # If yards increased but EP also increased (catch ALL cases)
                    if (curr_yards > prev_yards and curr_ep > prev_ep):
                        
                        ep_change = curr_ep - prev_ep
                        severity = 'high' if ep_change > 0.5 else 'medium' if ep_change > 0.1 else 'low'
                        
                        # Flag both the current and previous situation
                        df.loc[curr_idx, 'yards_anomaly'] = True
                        df.loc[curr_idx, 'yards_anomaly_severity'] = severity
                        df.loc[curr_idx, 'yards_anomaly_ep_change'] = ep_change
                        
                        df.loc[prev_idx, 'yards_anomaly'] = True
                        df.loc[prev_idx, 'yards_anomaly_severity'] = severity
                        df.loc[prev_idx, 'yards_anomaly_ep_change'] = ep_change
                        
                        anomaly_count += 1
                
                pbar.update(1)
        
        logging.info(f"Flagged {anomaly_count} yards_to_goal anomalies")
        return df
    
    def flag_down_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag situations where later downs have higher EP than earlier downs"""
        logging.info("Flagging down anomalies...")
        
        # Initialize anomaly flags
        df['down_anomaly'] = False
        df['down_anomaly_severity'] = 'none'
        df['down_anomaly_ep_change'] = 0.0
        
        # Group by time/distance/yards and check down trends
        groups = df.groupby(['half_seconds_remaining', 'distance', 'yards_to_goal'])
        
        anomaly_count = 0
        with tqdm(total=len(groups), desc="Checking down trends", unit="group") as pbar:
            for (time, dist, yards), group in groups:
                if len(group) < 2:
                    pbar.update(1)
                    continue
                
                # Sort by down
                group_sorted = group.sort_values('down')
                indices = group_sorted.index
                
                # Check each consecutive down pair
                for i in range(1, len(group_sorted)):
                    curr_idx = indices[i]
                    prev_idx = indices[i-1]
                    
                    curr_down = group_sorted.iloc[i]['down']
                    prev_down = group_sorted.iloc[i-1]['down']
                    curr_ep = group_sorted.iloc[i]['expected_points']
                    prev_ep = group_sorted.iloc[i-1]['expected_points']
                    
                    # Later downs having higher EP is suspicious (catch ALL cases)
                    ep_increase = curr_ep - prev_ep
                    if curr_down > prev_down and curr_ep > prev_ep:
                        severity = 'high' if (curr_down == 4 and prev_down == 3) else 'medium' if ep_increase > 0.1 else 'low'
                        
                        # Flag both situations
                        df.loc[curr_idx, 'down_anomaly'] = True
                        df.loc[curr_idx, 'down_anomaly_severity'] = severity
                        df.loc[curr_idx, 'down_anomaly_ep_change'] = ep_increase
                        
                        df.loc[prev_idx, 'down_anomaly'] = True
                        df.loc[prev_idx, 'down_anomaly_severity'] = severity
                        df.loc[prev_idx, 'down_anomaly_ep_change'] = ep_increase
                        
                        anomaly_count += 1
                
                pbar.update(1)
        
        logging.info(f"Flagged {anomaly_count} down anomalies")
        return df
    
    def flag_distance_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag situations where longer distances have higher EP"""
        logging.info("Flagging distance anomalies...")
        
        # Initialize anomaly flags
        df['distance_anomaly'] = False
        df['distance_anomaly_severity'] = 'none'
        df['distance_anomaly_ep_change'] = 0.0
        
        # Group by time/down/yards and check distance trends
        groups = df.groupby(['half_seconds_remaining', 'down', 'yards_to_goal'])
        
        anomaly_count = 0
        with tqdm(total=len(groups), desc="Checking distance trends", unit="group") as pbar:
            for (time, down, yards), group in groups:
                if len(group) < 2:
                    pbar.update(1)
                    continue
                
                # Sort by distance
                group_sorted = group.sort_values('distance')
                indices = group_sorted.index
                
                # Check each consecutive distance pair
                for i in range(1, len(group_sorted)):
                    curr_idx = indices[i]
                    prev_idx = indices[i-1]
                    
                    curr_dist = group_sorted.iloc[i]['distance']
                    prev_dist = group_sorted.iloc[i-1]['distance']
                    curr_ep = group_sorted.iloc[i]['expected_points']
                    prev_ep = group_sorted.iloc[i-1]['expected_points']
                    
                    dist_increase = curr_dist - prev_dist
                    ep_increase = curr_ep - prev_ep
                    
                    # Longer distance with higher EP is problematic (catch ALL cases)
                    if (dist_increase > 0 and ep_increase > 0):
                        severity = 'high' if ep_increase > 0.5 else 'medium' if ep_increase > 0.1 else 'low'
                        
                        # Flag both situations
                        df.loc[curr_idx, 'distance_anomaly'] = True
                        df.loc[curr_idx, 'distance_anomaly_severity'] = severity
                        df.loc[curr_idx, 'distance_anomaly_ep_change'] = ep_increase
                        
                        df.loc[prev_idx, 'distance_anomaly'] = True
                        df.loc[prev_idx, 'distance_anomaly_severity'] = severity
                        df.loc[prev_idx, 'distance_anomaly_ep_change'] = ep_increase
                        
                        anomaly_count += 1
                
                pbar.update(1)
        
        logging.info(f"Flagged {anomaly_count} distance anomalies")
        return df
    
    def smooth_yards_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling average smoothing to groups with yards anomalies"""
        logging.info("Smoothing yards_to_goal anomalies...")
        
        df_smoothed = df.copy()
        df_smoothed['expected_points_smoothed'] = df_smoothed['expected_points'].copy()
        smoothed_count = 0
        
        # Group by situation (excluding yards_to_goal)
        groups = df.groupby(['half_seconds_remaining', 'down', 'distance'])
        
        with tqdm(total=len(groups), desc="Smoothing anomalies", unit="group") as pbar:
            for (time, down, dist), group in groups:
                # Only process groups that have anomalies
                if not group['yards_anomaly'].any() or len(group) < 3:
                    pbar.update(1)
                    continue
                
                # Sort by yards_to_goal
                group_sorted = group.sort_values('yards_to_goal').copy()
                
                # Step 1: Apply rolling average
                window_size = min(5, len(group_sorted))
                smoothed_values = group_sorted['expected_points'].rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean()
                
                # Step 2: Enforce monotonic constraint - EP should decrease as yards increase
                # Convert to array for easier manipulation
                smoothed_array = smoothed_values.values
                
                for i in range(1, len(smoothed_array)):
                    # If current EP is higher than previous (violation), set it equal to previous
                    if smoothed_array[i] > smoothed_array[i-1]:
                        smoothed_array[i] = smoothed_array[i-1]
                
                # Step 3: Create incremental decreases for flat segments
                # Find consecutive equal values and interpolate between them
                i = 0
                while i < len(smoothed_array) - 1:
                    # Find the start and end of a flat segment
                    if smoothed_array[i] == smoothed_array[i + 1]:
                        start_idx = i
                        end_idx = i + 1
                        
                        # Find the full extent of the flat segment
                        while end_idx < len(smoothed_array) - 1 and smoothed_array[end_idx] == smoothed_array[end_idx + 1]:
                            end_idx += 1
                        
                        # Only interpolate if we have a segment of 2+ equal values
                        if end_idx > start_idx:
                            start_value = smoothed_array[start_idx]
                            
                            # Determine end value - either the next different value or a small decrease
                            if end_idx < len(smoothed_array) - 1:
                                end_value = smoothed_array[end_idx + 1]
                            else:
                                # Last segment - create small decreases
                                end_value = start_value - 0.05 * (end_idx - start_idx + 1)
                            
                            # Make sure end_value is not higher than start_value
                            if end_value >= start_value:
                                end_value = start_value - 0.01 * (end_idx - start_idx + 1)
                            
                            # Interpolate linearly between start and end
                            segment_length = end_idx - start_idx + 1
                            for j in range(segment_length):
                                weight = j / segment_length if segment_length > 1 else 0
                                smoothed_array[start_idx + j] = start_value + weight * (end_value - start_value)
                        
                        i = end_idx + 1
                    else:
                        i += 1
                
                # Update dataframe with final smoothed values
                df_smoothed.loc[group_sorted.index, 'expected_points_smoothed'] = smoothed_array
                
                # Count changes from original
                changes = abs(smoothed_array - group_sorted['expected_points'].values) > 0.001
                smoothed_count += changes.sum()
                
                pbar.update(1)
        
        # Add smoothing flag
        df_smoothed['was_smoothed'] = abs(df_smoothed['expected_points_smoothed'] - 
                                         df_smoothed['expected_points']) > 0.001
        
        logging.info(f"Smoothed {smoothed_count} data points")
        return df_smoothed
    
    def smooth_down_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to groups with down anomalies"""
        logging.info("Smoothing down anomalies...")
        
        df_smoothed = df.copy()
        if 'expected_points_smoothed' not in df_smoothed.columns:
            df_smoothed['expected_points_smoothed'] = df_smoothed['expected_points'].copy()
        
        smoothed_count = 0
        groups = df.groupby(['half_seconds_remaining', 'distance', 'yards_to_goal'])
        
        with tqdm(total=len(groups), desc="Smoothing down anomalies", unit="group") as pbar:
            for (time, dist, yards), group in groups:
                if not group['down_anomaly'].any() or len(group) < 2:
                    pbar.update(1)
                    continue
                
                group_sorted = group.sort_values('down').copy()
                
                # Step 1: Rolling average
                window_size = min(3, len(group_sorted))
                smoothed_values = group_sorted['expected_points_smoothed'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                
                # Step 2: Enforce monotonic constraint (EP should decrease as down increases)
                smoothed_array = smoothed_values.values
                for i in range(1, len(smoothed_array)):
                    if smoothed_array[i] > smoothed_array[i-1]:
                        smoothed_array[i] = smoothed_array[i-1]
                
                # Step 3: Interpolate flat segments
                i = 0
                while i < len(smoothed_array) - 1:
                    if smoothed_array[i] == smoothed_array[i + 1]:
                        start_idx = i
                        end_idx = i + 1
                        
                        while end_idx < len(smoothed_array) - 1 and smoothed_array[end_idx] == smoothed_array[end_idx + 1]:
                            end_idx += 1
                        
                        if end_idx > start_idx:
                            start_value = smoothed_array[start_idx]
                            
                            if end_idx < len(smoothed_array) - 1:
                                end_value = smoothed_array[end_idx + 1]
                            else:
                                end_value = start_value - 0.02 * (end_idx - start_idx + 1)
                            
                            if end_value >= start_value:
                                end_value = start_value - 0.01 * (end_idx - start_idx + 1)
                            
                            segment_length = end_idx - start_idx + 1
                            for j in range(segment_length):
                                weight = j / segment_length if segment_length > 1 else 0
                                smoothed_array[start_idx + j] = start_value + weight * (end_value - start_value)
                        
                        i = end_idx + 1
                    else:
                        i += 1
                
                df_smoothed.loc[group_sorted.index, 'expected_points_smoothed'] = smoothed_array
                changes = abs(smoothed_array - group_sorted['expected_points_smoothed'].values) > 0.001
                smoothed_count += changes.sum()
                pbar.update(1)
        
        logging.info(f"Smoothed {smoothed_count} down anomaly data points")
        return df_smoothed
    
    def smooth_distance_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to groups with distance anomalies"""
        logging.info("Smoothing distance anomalies...")
        
        df_smoothed = df.copy()
        if 'expected_points_smoothed' not in df_smoothed.columns:
            df_smoothed['expected_points_smoothed'] = df_smoothed['expected_points'].copy()
        
        smoothed_count = 0
        groups = df.groupby(['half_seconds_remaining', 'down', 'yards_to_goal'])
        
        with tqdm(total=len(groups), desc="Smoothing distance anomalies", unit="group") as pbar:
            for (time, down, yards), group in groups:
                if not group['distance_anomaly'].any() or len(group) < 2:
                    pbar.update(1)
                    continue
                
                group_sorted = group.sort_values('distance').copy()
                
                # Step 1: Rolling average
                window_size = min(3, len(group_sorted))
                smoothed_values = group_sorted['expected_points_smoothed'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                
                # Step 2: Enforce monotonic constraint (EP should decrease as distance increases)
                smoothed_array = smoothed_values.values
                for i in range(1, len(smoothed_array)):
                    if smoothed_array[i] > smoothed_array[i-1]:
                        smoothed_array[i] = smoothed_array[i-1]
                
                # Step 3: Interpolate flat segments
                i = 0
                while i < len(smoothed_array) - 1:
                    if smoothed_array[i] == smoothed_array[i + 1]:
                        start_idx = i
                        end_idx = i + 1
                        
                        while end_idx < len(smoothed_array) - 1 and smoothed_array[end_idx] == smoothed_array[end_idx + 1]:
                            end_idx += 1
                        
                        if end_idx > start_idx:
                            start_value = smoothed_array[start_idx]
                            
                            if end_idx < len(smoothed_array) - 1:
                                end_value = smoothed_array[end_idx + 1]
                            else:
                                end_value = start_value - 0.02 * (end_idx - start_idx + 1)
                            
                            if end_value >= start_value:
                                end_value = start_value - 0.01 * (end_idx - start_idx + 1)
                            
                            segment_length = end_idx - start_idx + 1
                            for j in range(segment_length):
                                weight = j / segment_length if segment_length > 1 else 0
                                smoothed_array[start_idx + j] = start_value + weight * (end_value - start_value)
                        
                        i = end_idx + 1
                    else:
                        i += 1
                
                df_smoothed.loc[group_sorted.index, 'expected_points_smoothed'] = smoothed_array
                changes = abs(smoothed_array - group_sorted['expected_points_smoothed'].values) > 0.001
                smoothed_count += changes.sum()
                pbar.update(1)
        
        logging.info(f"Smoothed {smoothed_count} distance anomaly data points")
        return df_smoothed
    
    def validate_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if smoothing fixed the violations"""
        logging.info("Validating smoothing results...")
        
        violations_after = 0
        groups = df.groupby(['half_seconds_remaining', 'down', 'distance'])
        
        for (time, down, dist), group in groups:
            if len(group) < 2:
                continue
                
            group_sorted = group.sort_values('yards_to_goal')
            
            for i in range(1, len(group_sorted)):
                curr_yards = group_sorted.iloc[i]['yards_to_goal']
                prev_yards = group_sorted.iloc[i-1]['yards_to_goal']
                curr_ep = group_sorted.iloc[i]['expected_points_smoothed']
                prev_ep = group_sorted.iloc[i-1]['expected_points_smoothed']
                
                if (curr_yards > prev_yards and curr_ep > prev_ep + 0.05):
                    violations_after += 1
        
        original_violations = self._count_violations(df, 'expected_points')
        
        print(f"   Original violations: {original_violations}")
        print(f"   Violations after smoothing: {violations_after}")
        print(f"   Violations removed: {original_violations - violations_after}")
        
        return df
    
    def _count_violations(self, df: pd.DataFrame, ep_column: str) -> int:
        """Count violations in the dataset"""
        violations = 0
        groups = df.groupby(['half_seconds_remaining', 'down', 'distance'])
        
        for (time, down, dist), group in groups:
            if len(group) < 2:
                continue
            group_sorted = group.sort_values('yards_to_goal')
            for i in range(1, len(group_sorted)):
                curr_yards = group_sorted.iloc[i]['yards_to_goal']
                prev_yards = group_sorted.iloc[i-1]['yards_to_goal']
                curr_ep = group_sorted.iloc[i][ep_column]
                prev_ep = group_sorted.iloc[i-1][ep_column]
                
                if (curr_yards > prev_yards and curr_ep > prev_ep + 0.05):
                    violations += 1
        
        return violations
    
    def create_summary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary anomaly flags"""
        logging.info("Creating summary flags...")
        
        df['any_anomaly'] = (df['yards_anomaly'] | 
                            df['down_anomaly'] | 
                            df['distance_anomaly'])
        
        df['anomaly_count'] = (df['yards_anomaly'].astype(int) + 
                              df['down_anomaly'].astype(int) + 
                              df['distance_anomaly'].astype(int))
        
        # Overall severity
        def get_max_severity(row):
            severities = [row['yards_anomaly_severity'], 
                         row['down_anomaly_severity'], 
                         row['distance_anomaly_severity']]
            
            if 'high' in severities:
                return 'high'
            elif 'medium' in severities:
                return 'medium'
            else:
                return 'none'
        
        df['overall_severity'] = df.apply(get_max_severity, axis=1)
        df['critical_anomaly'] = ((df['anomaly_count'] >= 2) | 
                                 (df['overall_severity'] == 'high'))
        
        return df
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate summary report"""
        
        report = []
        report.append("=" * 80)
        report.append("EXPECTED POINTS ANOMALY DETECTION & SMOOTHING REPORT")
        report.append("=" * 80)
        report.append(f"Total rows: {len(df):,}")
        report.append("")
        
        # Anomaly counts
        yards_anomalies = df['yards_anomaly'].sum()
        down_anomalies = df['down_anomaly'].sum()
        distance_anomalies = df['distance_anomaly'].sum()
        any_anomalies = df['any_anomaly'].sum()
        
        report.append("ANOMALY COUNTS:")
        report.append(f"Yards anomalies: {yards_anomalies:,}")
        report.append(f"Down anomalies: {down_anomalies:,}")
        report.append(f"Distance anomalies: {distance_anomalies:,}")
        report.append(f"Any anomaly: {any_anomalies:,}")
        
        if 'was_smoothed' in df.columns:
            smoothed = df['was_smoothed'].sum()
            report.append(f"Points smoothed: {smoothed:,}")
            
            # Effectiveness
            original_violations = self._count_violations(df, 'expected_points')
            smoothed_violations = self._count_violations(df, 'expected_points_smoothed')
            report.append("")
            report.append("SMOOTHING EFFECTIVENESS:")
            report.append(f"Violations before: {original_violations:,}")
            report.append(f"Violations after: {smoothed_violations:,}")
            report.append(f"Violations removed: {original_violations - smoothed_violations:,}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Detect and smooth EP anomalies')
    
    parser.add_argument('--input_file', '-i', 
                       default='expected_points_lookup_table.csv',
                       help='Input CSV file')
    
    parser.add_argument('--output_file', '-o', 
                       default='expected_points_lookup_table_flagged.csv',
                       help='Output CSV file')
    
    parser.add_argument('--ep_threshold', '-t', type=float, default=0.2,
                       help='EP threshold for anomaly detection')
    
    parser.add_argument('--apply_smoothing', action='store_true', default=True,
                       help='Apply smoothing')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print(f"Loading data from: {args.input_file}")
        
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"File not found: {args.input_file}")
        
        df = pd.read_csv(args.input_file)
        print(f"   Loaded {len(df):,} rows")
        
        # Check required columns
        required_cols = ['half_seconds_remaining', 'down', 'distance', 'yards_to_goal', 'expected_points']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Initialize detector
        detector = EPAnomalyDetector(ep_threshold=args.ep_threshold)
        
        print("\nDetecting anomalies...")
        df = detector.flag_yards_anomalies(df)
        df = detector.flag_down_anomalies(df)
        df = detector.flag_distance_anomalies(df)
        
        # Smoothing
        if args.apply_smoothing:
            print("Applying smoothing...")
            df = detector.smooth_yards_anomalies(df)
            df = detector.smooth_down_anomalies(df) 
            df = detector.smooth_distance_anomalies(df)
            df = detector.validate_smoothing(df)
        else:
            df['expected_points_smoothed'] = df['expected_points']
            df['was_smoothed'] = False
        
        # Summary
        df = detector.create_summary_flags(df)
        
        # Report
        report = detector.generate_report(df)
        print("\n" + report)
        
        # Save
        print(f"\nSaving to: {args.output_file}")
        df.to_csv(args.output_file, index=False)
        
        print("Complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())