import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from scipy import interpolate

def plot_first_and_ten_distance_vs_ep(df):
    """
    Filter to 1st and 10s and plot distance (yards to end zone) vs Expected Points
    """
    # Filter to only 1st and 10 situations
    first_and_ten = df[(df['down'] == 1) & (df['distance'] == 10)].copy()
    
    print(f"Found {len(first_and_ten)} rows with 1st and 10 situations")
    
    if len(first_and_ten) == 0:
        print("No 1st and 10 situations found in the data!")
        return
    
    # Sort by distance for better visualization
    first_and_ten = first_and_ten.sort_values('yards_to_goal')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(first_and_ten['yards_to_goal'], first_and_ten['expected_points'], 
                alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    if len(first_and_ten) > 1:
        z = np.polyfit(first_and_ten['yards_to_goal'], first_and_ten['expected_points'], 1)
        p = np.poly1d(z)
        plt.plot(first_and_ten['yards_to_goal'], 
                p(first_and_ten['yards_to_goal']), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope: {z[0]:.4f})')
    
    # Formatting
    plt.xlabel('Yards to End Zone', fontsize=12)
    plt.ylabel('Expected Points', fontsize=12)
    plt.title('Expected Points vs Field Position (1st & 10 Only)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics text
    correlation = first_and_ten['yards_to_goal'].corr(first_and_ten['expected_points'])
    stats_text = f'Correlation: {correlation:.3f}\nData Points: {len(first_and_ten)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Save the plot as PNG
    plt.tight_layout()
    plt.savefig('first_and_ten_distance_vs_ep.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print("✓ Plot saved as 'first_and_ten_distance_vs_ep.png'")
    
    # Print some validation info
    print(f"\nValidation Check:")
    print(f"Correlation coefficient: {correlation:.4f}")
    if correlation < -0.5:
        print("✓ GOOD: Strong negative correlation (closer to end zone = higher EP)")
    elif correlation < -0.2:
        print("⚠ MODERATE: Weak negative correlation")
    else:
        print("✗ CONCERNING: Correlation should be negative!")
    
    # Show min/max values
    min_distance = first_and_ten['yards_to_goal'].min()
    max_distance = first_and_ten['yards_to_goal'].max()
    ep_at_min = first_and_ten[first_and_ten['yards_to_goal'] == min_distance]['expected_points'].iloc[0]
    ep_at_max = first_and_ten[first_and_ten['yards_to_goal'] == max_distance]['expected_points'].iloc[0]
    
    print(f"\nField Position Analysis:")
    print(f"Closest to end zone ({min_distance} yards): EP = {ep_at_min:.3f}")
    print(f"Farthest from end zone ({max_distance} yards): EP = {ep_at_max:.3f}")
    print(f"EP difference: {ep_at_min - ep_at_max:.3f}")
    
    return first_and_ten

def plot_first_ten_with_time_color(df):
    """
    Plot 1st and 10s with time bucket shown as color intensity
    """
    # Filter to only 1st and 10 situations
    first_and_ten = df[(df['down'] == 1) & (df['distance'] == 10)].copy()
    
    if len(first_and_ten) == 0:
        print("No 1st and 10 situations found!")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create color map based on time
    norm = mcolors.Normalize(vmin=first_and_ten['half_seconds_remaining'].min(), 
                    vmax=first_and_ten['half_seconds_remaining'].max())
    colormap = cm.plasma
    
    # Scatter plot with time as color
    scatter = plt.scatter(first_and_ten['yards_to_goal'], 
                         first_and_ten['expected_points'],
                         c=first_and_ten['half_seconds_remaining'], 
                         cmap=colormap, norm=norm,
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Remaining (seconds)', rotation=270, labelpad=20)
    
    # Add overall trend line
    z = np.polyfit(first_and_ten['yards_to_goal'], first_and_ten['expected_points'], 1)
    p = np.poly1d(z)
    plt.plot(first_and_ten['yards_to_goal'], 
            p(first_and_ten['yards_to_goal']), 
            "red", linewidth=2, alpha=0.8, label=f'Overall trend (slope: {z[0]:.4f})')
    
    plt.xlabel('Yards to End Zone', fontsize=12)
    plt.ylabel('Expected Points', fontsize=12)
    plt.title('1st & 10: Expected Points vs Field Position (Color = Time)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('first_ten_with_time_color.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Time-colored plot saved as 'first_ten_with_time_color.png'")

def plot_ep_by_down(df, distance=10, yards_to_endzone=None):
    """
    Plot Expected Points by down for a specific distance (default 10 yards)
    """
    # Filter data
    if yards_to_endzone is None:
        # Use all field positions
        filtered_data = df[df['distance'] == distance].copy()
        title_suffix = f"(Distance: {distance} yards, All Field Positions)"
    else:
        # Use specific field position
        filtered_data = df[(df['distance'] == distance) & 
                          (df['yards_to_goal'] == yards_to_endzone)].copy()
        title_suffix = f"(Distance: {distance}, Field Position: {yards_to_endzone} yards)"
    
    if len(filtered_data) == 0:
        print(f"No data found for distance={distance}, yards_to_endzone={yards_to_endzone}")
        return
    
    print(f"Found {len(filtered_data)} records for analysis")
    print(f"Downs available: {sorted(filtered_data['down'].unique())}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Box plot showing distribution by down
    downs = sorted(filtered_data['down'].unique())
    down_data = [filtered_data[filtered_data['down'] == down]['expected_points'].values 
                 for down in downs]
    
    # Create box plot
    bp = plt.boxplot(down_data, labels=[f"{down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'}" 
                                       for down in downs],
                    patch_artist=True, notch=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(downs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean values as red diamonds
    means = [filtered_data[filtered_data['down'] == down]['expected_points'].mean() 
             for down in downs]
    plt.scatter(range(1, len(downs)+1), means, color='red', marker='D', s=100, 
               label='Mean EP', zorder=5)
    
    # Add value labels on the means
    for i, mean_val in enumerate(means):
        plt.text(i+1, mean_val + 0.1, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Down', fontsize=12)
    plt.ylabel('Expected Points', fontsize=12)
    plt.title(f'Expected Points by Down {title_suffix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add validation text
    validation_text = "Expected: 1st > 2nd > 3rd > 4th"
    if len(means) >= 2:
        is_decreasing = all(means[i] >= means[i+1] for i in range(len(means)-1))
        status = "✓ CORRECT" if is_decreasing else "✗ INCORRECT"
        validation_text += f"\nActual trend: {status}"
    
    plt.text(0.02, 0.98, validation_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgreen" if "CORRECT" in validation_text else "lightcoral", alpha=0.7))
    
    plt.tight_layout()
    filename = f'ep_by_down_dist{distance}{"_all_fields" if yards_to_endzone is None else f"_yard{yards_to_endzone}"}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Down analysis plot saved as '{filename}'")
    
    # Print detailed statistics
    print("\nDETAILED STATISTICS BY DOWN:")
    print("="*40)
    for down in downs:
        subset = filtered_data[filtered_data['down'] == down]
        print(f"{down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} Down:")
        print(f"  Count: {len(subset)}")
        print(f"  Mean EP: {subset['expected_points'].mean():.4f}")
        print(f"  Std Dev: {subset['expected_points'].std():.4f}")
        print(f"  Range: {subset['expected_points'].min():.3f} to {subset['expected_points'].max():.3f}")
        print()
    
    return filtered_data

def plot_down_progression_multiple_scenarios(df):
    """
    Plot down progression for multiple distance/field position scenarios
    """
    # Define scenarios to analyze
    scenarios = [
        {'distance': 10, 'yards_to_goal': 30, 'name': '10 yards, 30-yard line'},
        {'distance': 10, 'yards_to_goal': 50, 'name': '10 yards, 50-yard line'},
        {'distance': 5, 'yards_to_goal': 30, 'name': '5 yards, 30-yard line'},
        {'distance': 1, 'yards_to_goal': 30, 'name': '1 yard, 30-yard line'}
    ]
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, scenario in enumerate(scenarios):
        # Get data for this scenario
        subset = df[(df['distance'] == scenario['distance']) & 
                   (df['yards_to_goal'] == scenario['yards_to_goal'])]
        
        if len(subset) == 0:
            print(f"No data for scenario: {scenario['name']}")
            continue
        
        # Calculate mean EP by down
        down_means = []
        downs = sorted(subset['down'].unique())
        
        for down in downs:
            down_subset = subset[subset['down'] == down]
            if len(down_subset) > 0:
                down_means.append(down_subset['expected_points'].mean())
            else:
                down_means.append(np.nan)
        
        # Plot line for this scenario
        plt.plot(downs, down_means, marker='o', linewidth=2, markersize=8,
                color=colors[i % len(colors)], label=scenario['name'])
        
        # Add value labels
        for down, mean_ep in zip(downs, down_means):
            if not np.isnan(mean_ep):
                plt.text(down, mean_ep + 0.05, f'{mean_ep:.2f}', 
                        ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Down', fontsize=12)
    plt.ylabel('Mean Expected Points', fontsize=12)
    plt.title('Expected Points Progression by Down (Multiple Scenarios)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks([1, 2, 3, 4], ['1st', '2nd', '3rd', '4th'])
    
    plt.tight_layout()
    plt.savefig('down_progression_multiple_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Multiple scenarios plot saved as 'down_progression_multiple_scenarios.png'")

def analyze_down_effects_comprehensive(df):
    """
    Comprehensive analysis of down effects across all scenarios
    """
    print("COMPREHENSIVE DOWN EFFECTS ANALYSIS")
    print("="*50)
    
    # Group by distance and yards_to_goal, then analyze down effects
    grouped = df.groupby(['distance', 'yards_to_goal'])
    
    violations = []
    correct_progressions = 0
    total_scenarios = 0
    
    for (distance, yards), group in grouped:
        if len(group['down'].unique()) < 2:
            continue  # Skip if only one down available
        
        total_scenarios += 1
        
        # Calculate mean EP by down
        down_means = group.groupby('down')['expected_points'].mean().sort_index()
        
        # Check if it's decreasing
        is_decreasing = all(down_means.iloc[i] >= down_means.iloc[i+1] 
                           for i in range(len(down_means)-1))
        
        if is_decreasing:
            correct_progressions += 1
        else:
            violations.append({
                'distance': distance,
                'yards_to_goal': yards,
                'down_means': down_means.to_dict(),
                'violation_type': 'Non-decreasing EP progression'
            })
    
    print(f"Scenarios analyzed: {total_scenarios}")
    print(f"Correct progressions: {correct_progressions}")
    print(f"Violations: {len(violations)}")
    print(f"Success rate: {correct_progressions/total_scenarios*100:.1f}%")
    
    # Show worst violations
    if violations:
        print(f"\nTOP VIOLATIONS (showing first 5):")
        print("-"*40)
        for i, violation in enumerate(violations[:5]):
            print(f"{i+1}. Distance: {violation['distance']}, Yards: {violation['yards_to_goal']}")
            print(f"   Down EPs: {violation['down_means']}")
            print()
    
    return violations

def plot_heatmap_down_effects(df):
    """
    Create heatmap showing EP by down and field position
    """
    # Focus on 10-yard distance for simplicity
    subset = df[df['distance'] == 10].copy()
    
    if len(subset) == 0:
        print("No 10-yard distance data found for heatmap")
        return
    
    # Create pivot table
    pivot_data = subset.groupby(['down', 'yards_to_goal'])['expected_points'].mean().reset_index()
    heatmap_data = pivot_data.pivot(index='down', columns='yards_to_goal', values='expected_points')
    
    plt.figure(figsize=(16, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=False, fmt='.2f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Expected Points'})
    
    plt.title('Expected Points Heatmap: Down vs Field Position (10-yard distance)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Yards to End Zone', fontsize=12)
    plt.ylabel('Down', fontsize=12)
    
    # Customize y-axis labels
    plt.gca().set_yticklabels(['1st', '2nd', '3rd', '4th'][:len(heatmap_data.index)])
    
    plt.tight_layout()
    plt.savefig('down_field_position_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Down/field position heatmap saved as 'down_field_position_heatmap.png'")

def comprehensive_down_analysis(df):
    """
    Run all down-based analyses
    """
    print("Running comprehensive down analysis...")
    print("="*50)
    
    # Basic down analysis for different scenarios
    plot_ep_by_down(df, distance=10, yards_to_endzone=30)
    plot_ep_by_down(df, distance=10)  # All field positions
    
    # Multiple scenario comparison
    plot_down_progression_multiple_scenarios(df)
    
    # Heatmap
    plot_heatmap_down_effects(df)
    
    # Comprehensive validation
    violations = analyze_down_effects_comprehensive(df)
    
    print("\nFILES CREATED:")
    print("• ep_by_down_dist10_yard30.png - Down analysis for specific scenario")
    print("• ep_by_down_dist10_all_fields.png - Down analysis across all field positions")
    print("• down_progression_multiple_scenarios.png - Multiple scenario comparison")
    print("• down_field_position_heatmap.png - Heatmap of EP by down/field position")
    
    return violations

def recreate_cfbfastr_ep_by_down_graph(df, save_filename='ep_by_down_field_position.png'):
    """
    Recreate the cfbfastr Expected Points by Down and Field Position graph
    
    Args:
        df: DataFrame with columns ['down', 'distance', 'yards_to_goal', 'expected_points']
        save_filename: Name of file to save the plot
    """
    
    print("Recreating cfbfastr-style Expected Points by Down graph...")
    print(f"Using {len(df):,} situations from your lookup table")
    
    # Check required columns
    required_cols = ['down', 'distance', 'yards_to_goal', 'expected_points']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter to valid data
    valid_data = df[
        (df['down'].between(1, 4)) &
        (df['distance'].between(1, 99)) &
        (df['yards_to_goal'].between(1, 99)) &
        (~df['expected_points'].isna())
    ].copy()
    
    print(f"Using {len(valid_data):,} valid situations after filtering")
    
    # Create 5-yard bins for field position (like cfbfastr)
    valid_data['yardline_bin'] = (valid_data['yards_to_goal'] // 5) * 5
    valid_data['yardline_bin'] = valid_data['yardline_bin'].clip(5, 95)  # 5, 10, 15, ..., 95
    
    # Calculate mean EP by down and field position
    ep_by_down_field = valid_data.groupby(['down', 'yardline_bin'])['expected_points'].agg(['mean', 'count']).reset_index()
    ep_by_down_field.columns = ['down', 'yardline_bin', 'mean_ep', 'count']
    
    # For lookup tables, we typically have fewer observations per scenario
    # Adjust the minimum observations threshold
    min_observations = 10  # Reduced from 50 since lookup tables are more sparse
    ep_by_down_field = ep_by_down_field[ep_by_down_field['count'] >= min_observations]
    
    print("Sample size by down:")
    for down in [1, 2, 3, 4]:
        down_data = ep_by_down_field[ep_by_down_field['down'] == down]
        print(f"  {down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} down: {len(down_data)} field position bins")
    
    # Create the plot with cfbfastr styling
    plt.figure(figsize=(12, 8))
    
    # Define colors and styles to match cfbfastr
    colors = {
        1: '#2E8B57',  # Sea Green (1st down)
        2: '#FF8C00',  # Dark Orange (2nd down)  
        3: '#9370DB',  # Medium Purple (3rd down)
        4: '#DC143C'   # Crimson (4th down)
    }
    
    line_styles = {1: '-', 2: '--', 3: '-.', 4: ':'}
    
    # Plot each down
    for down in [1, 2, 3, 4]:
        down_data = ep_by_down_field[ep_by_down_field['down'] == down].copy()
        
        if len(down_data) == 0:
            print(f"Warning: No data for {down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} down")
            continue
        
        # Sort by field position
        down_data = down_data.sort_values('yardline_bin')
        
        x = down_data['yardline_bin']
        y = down_data['mean_ep']
        
        # Create smooth curve using interpolation (like cfbfastr)
        if len(x) >= 4:  # Need at least 4 points for smooth interpolation
            # Extend range to full field for interpolation
            x_smooth = np.arange(5, 100, 2.5)  # Every 2.5 yards
            
            # Use cubic spline interpolation for smooth curves
            try:
                f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
                y_smooth = f(x_smooth)
                
                # Plot smooth line
                plt.plot(x_smooth, y_smooth, 
                        color=colors[down], 
                        linewidth=2.5, 
                        linestyle=line_styles[down],
                        label=f"{down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'}")
                
                # Add scatter points for actual data
                plt.scatter(x, y, color=colors[down], s=20, alpha=0.7, zorder=5)
                
            except Exception as e:
                print(f"Interpolation failed for {down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} down: {e}")
                # Fallback to simple line plot
                plt.plot(x, y, color=colors[down], linewidth=2.5, marker='o', 
                        label=f"{down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'}")
        else:
            # Simple line plot for insufficient data
            plt.plot(x, y, color=colors[down], linewidth=2.5, marker='o',
                    label=f"{down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'}")
    
    # Formatting to match cfbfastr style
    plt.xlim(100, 0)  # Reverse x-axis like cfbfastr
    plt.xlabel('Yards from Opponent\'s End Zone', fontsize=12, fontweight='bold')
    plt.ylabel('Expected Points', fontsize=12, fontweight='bold')
    plt.title('Relationship between Field Position and Expected Points by Down\nYour Polynomial Model vs cfbfastr Style', 
              fontsize=14, fontweight='bold')
    
    # Add red zone shading (like cfbfastr)
    plt.axvspan(0, 20, alpha=0.2, color='red', label='Red Zone')
    plt.text(10, plt.ylim()[1] * 0.9, 'Red\nZone', fontsize=12, fontweight='bold', 
             ha='center', va='center', color='darkred')
    
    # Grid styling
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis to match typical EP range
    plt.ylim(-2, 7)
    
    # Add subtle background color
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Graph saved as '{save_filename}'")
    
    # Print validation summary
    print("\nVALIDATION SUMMARY:")
    print("="*30)
    
    # Check if progression is correct at key field positions
    test_positions = [50, 30, 20, 10]  # Midfield, 30-yard line, 20-yard line, 10-yard line
    
    for pos in test_positions:
        print(f"\nAt {pos}-yard line:")
        pos_data = ep_by_down_field[
            (ep_by_down_field['yardline_bin'] >= pos - 2.5) & 
            (ep_by_down_field['yardline_bin'] <= pos + 2.5)
        ]
        
        if len(pos_data) > 0:
            down_eps = pos_data.set_index('down')['mean_ep'].sort_index()
            
            for down, ep in down_eps.items():
                print(f"  {down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} down: {ep:.3f} EP")
            
            # Check if progression is logical
            if len(down_eps) >= 2:
                is_decreasing = all(down_eps.iloc[i] >= down_eps.iloc[i+1] 
                                  for i in range(len(down_eps)-1))
                status = "✓ CORRECT" if is_decreasing else "✗ WRONG"
                print(f"  Progression: {status}")
    
    return ep_by_down_field

def main():
    """
    Main function to run all analyses
    """
    # Replace with your actual lookup table file
    lookup_file = 'expected_points_lookup_table.csv'
    
    try:
        print("CFBFASTR-STYLE POLYNOMIAL MODEL ANALYSIS")
        print("="*50)
        
        # Load data
        df = pd.read_csv(lookup_file)
        print(f"✓ Loaded lookup table: {lookup_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Expected Points range: {df['expected_points'].min():.3f} to {df['expected_points'].max():.3f}")
        
        # Run basic field position analysis
        print("\n1. FIELD POSITION ANALYSIS (1st & 10)")
        print("-" * 40)
        first_ten_data = plot_first_and_ten_distance_vs_ep(df)
        
        # Run time-colored analysis
        print("\n2. TIME EFFECT ANALYSIS")
        print("-" * 40)
        plot_first_ten_with_time_color(df)
        
        # Run comprehensive down analysis
        print("\n3. DOWN PROGRESSION ANALYSIS")
        print("-" * 40)
        violations = comprehensive_down_analysis(df)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("\nFiles created:")
        print("• first_and_ten_distance_vs_ep.png")
        print("• first_ten_with_time_color.png")
        print("• ep_by_down_dist10_yard30.png")
        print("• ep_by_down_dist10_all_fields.png")
        print("• down_progression_multiple_scenarios.png")
        print("• down_field_position_heatmap.png")
        print("• your_model_ep_by_down.png")
        
        print(f"\nModel Validation Summary:")
        print(f"• Down progression violations: {len(violations) if violations else 0}")
        print(f"• Lookup table completeness: {len(df):,} situations")
        
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Could not find {lookup_file}")
        print("\nMake sure you've run the polynomial model training first!")
        print("The file should have these columns:")
        print("- down (1,2,3,4)")
        print("- distance (yards to go)")
        print("- yards_to_goal (field position)")
        print("- expected_points (your model's EP values)")
        print("- half_seconds_remaining (time)")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS: All analyses completed!")
        print("Check the generated PNG files to validate your polynomial model.")
    else:
        print("\n❌ FAILED: Analysis could not be completed.")