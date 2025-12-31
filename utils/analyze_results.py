import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_folder_name(folder_name):
    """
    Parses 'recon_flowdps_st50_a60_s0.5' into dictionary parameters.
    """
    try:
        # Regex to capture parameters
        # Matches: recon_{method}_st{steps}_a{angles}_s{scale}
        match = re.search(r"recon_([a-zA-Z]+)_st(\d+)_a(\d+)_s([\d.]+)", folder_name)
        if match:
            return {
                "Method": match.group(1),
                "Steps": int(match.group(2)),
                "Angles": int(match.group(3)),
                "Scale": float(match.group(4)),
            }
    except Exception as e:
        pass
    return None


def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to main experiment folder (e.g. experiments/mayo_fm_test)",
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=None,
        help="Filter experiments by number of angles (e.g. 60, 180). If not set, use all.",
    )

    args = parser.parse_args()

    results = []

    # 1. Walk through all subdirectories
    subdirs = glob.glob(os.path.join(args.exp_dir, "recon_*"))

    print(f"Found {len(subdirs)} experiments. Parsing...")

    for subdir in subdirs:
        folder_name = os.path.basename(subdir)
        params = parse_folder_name(folder_name)

        metrics_path = os.path.join(subdir, "metrics.csv")

        if params and os.path.exists(metrics_path):
            try:
                # Read CSV
                df = pd.read_csv(metrics_path)
                # Get the last row (Final result)
                final_metrics = df.iloc[-1]

                # Combine info
                entry = params.copy()
                entry["PSNR"] = final_metrics["PSNR"]
                entry["SSIM"] = final_metrics["SSIM"]
                entry["LPIPS"] = final_metrics["LPIPS"]
                results.append(entry)
            except Exception as e:
                print(f"Error reading {subdir}: {e}")

    if not results:
        print("No valid results found.")
        return

    # 2. Create Master DataFrame
    master_df = pd.DataFrame(results)

    # ---- Optional filtering by number of angles ----
    if args.angles is not None:
        master_df = master_df[master_df["Angles"] == args.angles]

        if master_df.empty:
            print(f"No results found for Angles = {args.angles}")
            return

        print(f"Filtered results: Angles = {args.angles}")

    # Save Master CSV
    output_csv = os.path.join(args.exp_dir, "grid_search_summary.csv")
    master_df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")

    # 3. Visualization (Heatmaps)
    # We create one heatmap per Method
    methods = master_df["Method"].unique()

    sns.set_theme(style="whitegrid")

    for method in methods:
        subset = master_df[master_df["Method"] == method]

        # Check if we have enough data to pivot (Steps x Scale)
        if len(subset["Steps"].unique()) > 1 and len(subset["Scale"].unique()) > 1:

            # Pivot: Rows=Steps, Cols=Scale, Values=PSNR
            pivot_table = subset.pivot(index="Steps", columns="Scale", values="PSNR")

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar_kws={"label": "PSNR (dB)"},
            )
            angle_str = f" | Angles={args.angles}" if args.angles is not None else ""
            plt.title(f"Performance Heatmap - {method.upper()}{angle_str}")
            plt.ylabel("Number of Steps")
            plt.xlabel("Guidance Scale")

            suffix = f"_a{args.angles}" if args.angles is not None else ""
            save_path = os.path.join(args.exp_dir, f"heatmap_{method}{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Heatmap saved: {save_path}")
        else:
            print(
                f"Not enough variation to generate heatmap for {method} (Needs multiple steps & scales)."
            )

    # 4. Global Comparison (Best per method)
    print("\n--- Best Configuration per Method (by PSNR) ---")
    best_df = master_df.loc[master_df.groupby("Method")["PSNR"].idxmax()]
    print(best_df[["Method", "Steps", "Scale", "PSNR", "SSIM", "LPIPS"]])


if __name__ == "__main__":
    analyze()
