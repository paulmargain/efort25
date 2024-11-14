import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

def load_and_process_data(file_path):
    data = np.load(file_path)
    # Create mask for invalid values
    mask = data == -1
    # Set invalid values to nan for better visualization
    data = data.astype(np.float32)  # Ensure data is float for NaNs
    data[mask] = np.nan
    return data

def create_comparison_plot(input_data, target_diff, pred_diff, title):
    # Compute real and synthetic follow-up maps
    real_baseline = input_data + target_diff
    synthetic_baseline = input_data + pred_diff

    # Compute metrics between synthetic_baseline and real_baseline
    real_baseline_flat = real_baseline.flatten()
    synthetic_baseline_flat = synthetic_baseline.flatten()
    input_data_flat = input_data.flatten()
    mask_sr = ~np.isnan(real_baseline_flat) & ~np.isnan(synthetic_baseline_flat)
    mask_input = ~np.isnan(real_baseline_flat) & ~np.isnan(input_data_flat)

    if np.sum(mask_sr) > 0:
        # Compute Spearman correlation and MAE for synthetic vs real
        spearman_corr_sr, _ = spearmanr(real_baseline_flat[mask_sr], synthetic_baseline_flat[mask_sr])
        mae_sr = np.mean(np.abs(real_baseline_flat[mask_sr] - synthetic_baseline_flat[mask_sr]))
    else:
        spearman_corr_sr = np.nan
        mae_sr = np.nan

    if np.sum(mask_input) > 0:
        # Compute Spearman correlation and MAE for input vs real
        spearman_corr_input, _ = spearmanr(real_baseline_flat[mask_input], input_data_flat[mask_input])
        mae_input = np.mean(np.abs(real_baseline_flat[mask_input] - input_data_flat[mask_input]))
    else:
        spearman_corr_input = np.nan
        mae_input = np.nan

    # Set up subplots with black background
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.patch.set_facecolor('black')
    fig.suptitle(title, fontsize=20, color='white')

    # Remove the middle axis in the third row as it's not used
    axes[2,1].axis('off')

    # Set common colormaps
    thickness_cmap = 'jet'  # Use perceptually uniform colormap
    diff_cmap = 'bwr'  # For differences

    # Find common value ranges for consistent colorbars
    thickness_vmin = np.nanmin([real_baseline, synthetic_baseline, input_data])
    thickness_vmax = np.nanmax([real_baseline, synthetic_baseline, input_data])

    diff_vmin = np.nanmin([target_diff, pred_diff])
    diff_vmax = np.nanmax([target_diff, pred_diff])

    # First row: Input Data, Target Difference, Real Baseline
    im0 = axes[0,0].imshow(input_data, cmap=thickness_cmap, vmin=thickness_vmin, vmax=thickness_vmax)
    axes[0,0].set_title('Input Data', fontsize=14, color='white')
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(target_diff, cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
    axes[0,1].set_title('Target Difference', fontsize=14, color='white')
    fig.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(real_baseline, cmap=thickness_cmap, vmin=thickness_vmin, vmax=thickness_vmax)
    axes[0,2].set_title('Real Baseline (Input + Target Diff)', fontsize=14, color='white')
    fig.colorbar(im2, ax=axes[0,2])

    # Second row: Input Data, Predicted Difference, Synthetic Baseline
    im3 = axes[1,0].imshow(input_data, cmap=thickness_cmap, vmin=thickness_vmin, vmax=thickness_vmax)
    axes[1,0].set_title('Input Data', fontsize=14, color='white')
    fig.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(pred_diff, cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
    axes[1,1].set_title('Predicted Difference', fontsize=14, color='white')
    fig.colorbar(im4, ax=axes[1,1])

    im5 = axes[1,2].imshow(synthetic_baseline, cmap=thickness_cmap, vmin=thickness_vmin, vmax=thickness_vmax)
    axes[1,2].set_title('Synthetic Baseline (Input + Predicted Diff)', fontsize=14, color='white')
    fig.colorbar(im5, ax=axes[1,2])

    # Third row: Correlation plots
    # Correlation plot 1: Synthetic Baseline vs Real Baseline
    axes[2,0].scatter(real_baseline_flat[mask_sr], synthetic_baseline_flat[mask_sr], alpha=0.5, s=5, color='white')
    # Plot y = x line
    min_val = np.nanmin([real_baseline_flat[mask_sr], synthetic_baseline_flat[mask_sr]])
    max_val = np.nanmax([real_baseline_flat[mask_sr], synthetic_baseline_flat[mask_sr]])
    axes[2,0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[2,0].set_xlabel('Real Baseline', fontsize=12, color='white')
    axes[2,0].set_ylabel('Synthetic Baseline', fontsize=12, color='white')
    axes[2,0].set_title('Synthetic vs Real Baseline', fontsize=14, color='white')
    axes[2,0].text(0.05, 0.95, f'Spearman r: {spearman_corr_sr:.2f}\nMAE: {mae_sr:.2f} mm',
                   transform=axes[2,0].transAxes, fontsize=20, verticalalignment='top', 
                   bbox=dict(facecolor='black', alpha=0.5), color='white')
    # Set ticks color to white
    axes[2,0].tick_params(colors='white')

    # Correlation plot 2: Input Data vs Real Baseline
    axes[2,2].scatter(real_baseline_flat[mask_input], input_data_flat[mask_input], alpha=0.5, s=5, color='white')
    # Plot y = x line
    min_val_input = np.nanmin([real_baseline_flat[mask_input], input_data_flat[mask_input]])
    max_val_input = np.nanmax([real_baseline_flat[mask_input], input_data_flat[mask_input]])
    axes[2,2].plot([min_val_input, max_val_input], [min_val_input, max_val_input], color='red', linestyle='--')
    axes[2,2].set_xlabel('Real Baseline', fontsize=12, color='white')
    axes[2,2].set_ylabel('Input Data', fontsize=12, color='white')
    axes[2,2].set_title('Input vs Real Baseline', fontsize=14, color='white')
    axes[2,2].text(0.05, 0.95, f'Spearman r: {spearman_corr_input:.2f}\nMAE: {mae_input:.2f} mm',
                   transform=axes[2,2].transAxes, fontsize=20, verticalalignment='top',
                   bbox=dict(facecolor='black', alpha=0.5), color='white')
    # Set ticks color to white
    axes[2,2].tick_params(colors='white')

    # Find global min and max for both plots
    global_min = min(min_val, min_val_input)
    global_max = max(max_val, max_val_input)

    # Set same range for both plots
    axes[2,0].set_xlim(global_min, global_max)
    axes[2,0].set_ylim(global_min, global_max)
    axes[2,2].set_xlim(global_min, global_max)
    axes[2,2].set_ylim(global_min, global_max)

    # Remove ticks from all images and set background color for all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')

    # Adjust layout
    plt.tight_layout()
    return fig

def main():
    # Add file uploader for zip file
    uploaded_file = st.file_uploader("Upload test data zip file", type="zip")
    
    if not uploaded_file:
        st.warning("Please upload a zip file containing the test data folders")
        return
        
    # Create temp directory and extract zip
    import tempfile
    import zipfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded file
        zip_path = temp_path / "test_data.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
            
        # Define paths relative to extracted content
        base_dir = temp_path / "test_data"
        test_results_dir = base_dir / "test_results" 
        test_inputs_dir = base_dir / "test_inputs"
        test_outputs_dir = base_dir / "test_outputs"
        
        # Get all prediction files
        pred_files = list(test_results_dir.glob("*_prediction.npy"))
        
        # Filter for KL >= 2 and valid KL values
        kl_filtered_files = []
        for pred_file in pred_files:
            try:
                # Extract KL grade from filename (assuming format: ID_SIDE_KL_...)
                kl_grade = pred_file.stem.split('_')[2]
                # Skip if KL is 'nan' or not a number
                if kl_grade.lower() == 'nan' or not kl_grade.isdigit():
                    continue
                kl_grade = int(kl_grade)
                if kl_grade >= 2:
                    kl_filtered_files.append(pred_file)
            except (IndexError, ValueError):
                continue
        
        # Sort files by ID and timepoint
        kl_filtered_files.sort()
        
        if not kl_filtered_files:
            st.error("No files found with KL grade >= 2.")
            return
        
        # Create selectbox for file selection
        selected_file = st.selectbox(
            "Select a subject",
            kl_filtered_files,
            format_func=lambda x: x.stem
        )
        
        if selected_file:
            # Get corresponding input and target files
            base_name = selected_file.stem.replace("_prediction", "")
            input_file = test_inputs_dir / f"{base_name}.npy"
            
            # Extract ID and SIDE for output file matching
            id_side = "_".join(base_name.split('_')[:2])  # Get just ID_SIDE
            # Look for output file with any KL grade
            output_files = list(test_outputs_dir.glob(f"{id_side}_*_CTh_map_00m.npy"))
            
            if not output_files:
                st.error(f"No corresponding output file found for {id_side}")
                return
            
            output_file = output_files[0]  # Take the first matching output file
            
            # Load data
            input_data = load_and_process_data(input_file)
            target_data = load_and_process_data(output_file)
            pred_diff = load_and_process_data(selected_file)
            
            target_diff = target_data - input_data
            
            # Create and display plot
            fig = create_comparison_plot(
                input_data,
                target_diff,
                pred_diff,
                None
            )
            st.pyplot(fig)
        
        else:
            st.error("No file selected.")

if __name__ == "__main__":
    main()
