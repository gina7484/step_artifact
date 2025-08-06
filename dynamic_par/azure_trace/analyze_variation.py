import pandas as pd
import numpy as np


def analyze_context_tokens_std(input_file, output_file, batch_size):
    """
    Analyzes ContextTokens column and calculates standard deviations
    for sliding windows of size 64 starting at odd indices.
    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Calculate standard deviation of entire ContextTokens column
    overall_std = df["ContextTokens"].std()

    # Prepare results list
    results = []

    # Generate sliding windows: 1~64, 3~66, 5~68, ..., 4987~5050
    # Starting indices: 1, 3, 5, 7, ..., 4987 (odd numbers)
    start_indices = range(1, 4988, 2)  # 1, 3, 5, ..., 4987

    for start_idx in start_indices:
        end_idx = (
            start_idx + batch_size - 1
        )  # Window size of 64 (start_idx to start_idx+63 inclusive)

        # Check if we have enough data for this range
        if end_idx < len(df):
            # Extract the range (converting from 1-based to 0-based indexing)
            range_data = df["ContextTokens"].iloc[
                start_idx - 1 : end_idx
            ]  # -1 for 0-based indexing

            # Calculate standard deviation for this range
            range_std = range_data.std()

            # Store the result
            results.append(
                {
                    "overall_std_dev": overall_std,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "range_std_dev": range_std,
                }
            )
        else:
            # If we don't have enough data, break the loop
            break

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_file, index=False)

    print(f"Analysis complete!")
    print(f"Overall standard deviation of ContextTokens: {overall_std:.4f}")
    print(f"Generated {len(results)} sliding window calculations")
    print(f"Results saved to: {output_file}")

    return results_df


# Usage example
if __name__ == "__main__":
    # Replace 'input.csv' and 'output.csv' with your actual file names
    input_filename = "/home/ginasohn/step_tl/dynamic_par/azure_trace/AzureLLMInferenceTrace_conv_1_5000.csv"
    batch_size = 80
    output_filename = f"/home/ginasohn/step_tl/dynamic_par/azure_trace/std_analysis_results_b{batch_size}.csv"

    try:
        results = analyze_context_tokens_std(
            input_filename, output_filename, batch_size
        )

        # Display first few results
        print("\nFirst 5 results:")
        print(results.head())

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_filename}'")
        print("Please make sure the file exists in the current directory")
    except KeyError:
        print("Error: 'ContextTokens' column not found in the CSV file")
        print("Please check that the column name is correct")
    except Exception as e:
        print(f"An error occurred: {e}")
