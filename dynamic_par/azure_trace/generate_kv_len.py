import pandas as pd
import numpy as np


def extract_context_tokens(input_file, output_file, start_index, end_index, batch_size):
    """
    Extract ContextTokens for rows within specified index range and save as numpy array.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output .npy file
        start_index (int): Starting index (inclusive)
        end_index (int): Ending index (inclusive)
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Filter rows where Index is within the specified range
    filtered_df = df[(df["Index"] >= start_index) & (df["Index"] <= end_index)]

    # Extract only the ContextTokens column and convert to numpy array
    context_tokens = filtered_df["ContextTokens"].values

    # Ensure the array has exactly batch_size elements
    if len(context_tokens) > batch_size:
        # Truncate to first batch_size elements
        context_tokens = context_tokens[:batch_size]
        print(f"Warning: Truncated {len(filtered_df)} values to {batch_size} elements")
    elif len(context_tokens) < batch_size:
        # Pad with zeros to reach batch_size elements
        padding = batch_size - len(context_tokens)
        context_tokens = np.pad(
            context_tokens, (0, padding), "constant", constant_values=0
        )
        print(
            f"Padded {len(filtered_df)} values with {padding} zeros to reach {batch_size} elements"
        )

    # Save as numpy array with shape [batch_size]
    np.save(output_file, context_tokens)

    print(f"Saved numpy array of shape {context_tokens.shape} to {output_file}")
    print(f"Range: Index {start_index} to {end_index}")


# Example usage:
if __name__ == "__main__":
    # Example: extract ContextTokens for indices 2 to 4
    input_filename = "./dynamic_par/azure_trace/AzureLLMInferenceTrace_conv_1_5000.csv"

    # batch_size = 64
    # batch_list_b64 = [
    #     # batch with high stdev
    #     {"start": 961, "end": 1024, "stdev": 1457},
    #     {"start": 3239, "end": 3302, "stdev": 1374},
    #     {"start": 1727, "end": 1790, "stdev": 1370},
    #     # batch with medium_high stdev
    #     {"start": 3897, "end": 3960, "stdev": 1226},
    #     {"start": 881, "end": 944, "stdev": 1226},
    #     {"start": 1013, "end": 1076, "stdev": 1226},
    #     # batch with similar stdev
    #     {"start": 4007, "end": 4070, "stdev": 996},
    #     {"start": 3123, "end": 3186, "stdev": 995},
    #     {"start": 733, "end": 796, "stdev": 995},
    #     # batch with medium_low stdev
    #     {"start": 1505, "end": 1568, "stdev": 754},
    #     {"start": 355, "end": 418, "stdev": 754},
    #     {"start": 181, "end": 244, "stdev": 754},
    #     # batch with lowest stdev
    #     {"start": 2019, "end": 2082, "stdev": 508},
    #     {"start": 4185, "end": 4248, "stdev": 484},
    #     {"start": 271, "end": 334, "stdev": 477},
    # ]
    # for batch in batch_list_b64:
    #     extract_context_tokens(
    #         input_filename,
    #         f"./dynamic_par/azure_trace/b{batch_size}/conv_stdev{batch['stdev']:04d}_{batch['start']:04d}_{batch['end']:04d}.npy",
    #         batch["start"],
    #         batch["end"],
    #         batch_size,
    #     )

    # batch_size = 16
    # batch_list_b16 = [
    #     # batch with super high stdev due to the small batch size
    #     {"start": 1487, "end": 1502, "stdev": 1974},
    #     {"start": 3727, "end": 3742, "stdev": 1951},
    #     {"start": 1501, "end": 1516, "stdev": 1884},
    #     # batch with high stdev
    #     {"start": 2805, "end": 2820, "stdev": 1456},
    #     {"start": 821, "end": 836, "stdev": 1455},
    #     {"start": 985, "end": 1000, "stdev": 1454},
    #     # batch with medium_high stdev
    #     {"start": 4099, "end": 4114, "stdev": 1226},
    #     {"start": 1049, "end": 1064, "stdev": 1226},
    #     {"start": 75, "end": 90, "stdev": 1226},
    #     # batch with similar stdev
    #     {"start": 3349, "end": 3364, "stdev": 987},
    #     {"start": 3275, "end": 3290, "stdev": 987},
    #     {"start": 3181, "end": 3196, "stdev": 986},
    #     # batch with medium_low stdev
    #     {"start": 1451, "end": 1466, "stdev": 754},
    #     {"start": 1851, "end": 1866, "stdev": 754},
    #     {"start": 1989, "end": 2004, "stdev": 752},
    #     # batch with lowest stdev
    #     {"start": 3799, "end": 3814, "stdev": 480},
    #     {"start": 2477, "end": 2492, "stdev": 479},
    #     {"start": 1063, "end": 1078, "stdev": 479},
    #     # batch with super low stdev due to the small batch size
    #     {"start": 1683, "end": 1698, "stdev": 198},
    #     {"start": 1443, "end": 1458, "stdev": 198},
    #     {"start": 1845, "end": 1860, "stdev": 174},
    # ]
    # for batch in batch_list_b16:
    #     extract_context_tokens(
    #         input_filename,
    #         f"./dynamic_par/azure_trace/b{batch_size}/conv_stdev{batch['stdev']:04d}_{batch['start']:04d}_{batch['end']:04d}.npy",
    #         batch["start"],
    #         batch["end"],
    #         batch_size,
    #     )

    batch_size = 80
    batch_list_b80 = [
        # batch with high stdev
        {"start": 981, "end": 1060, "stdev": 1413},
        {"start": 3227, "end": 3306, "stdev": 1334},
        {"start": 815, "end": 894, "stdev": 1310},
        # batch with medium_high stdev
        {"start": 3537, "end": 3616, "stdev": 1226},
        {"start": 4059, "end": 4138, "stdev": 1225},
        {"start": 869, "end": 948, "stdev": 1224},
        # batch with similar stdev
        {"start": 4687, "end": 4766, "stdev": 987},
        {"start": 3989, "end": 4068, "stdev": 986},
        {"start": 1891, "end": 1970, "stdev": 987},
        # batch with medium_low stdev
        {"start": 309, "end": 388, "stdev": 755},
        {"start": 2115, "end": 2194, "stdev": 754},
        {"start": 101, "end": 180, "stdev": 754},
        # batch with lowest stdev
        {"start": 4181, "end": 4260, "stdev": 562},
        {"start": 135, "end": 214, "stdev": 531},
        {"start": 2025, "end": 2104, "stdev": 529},
    ]
    for batch in batch_list_b80:
        extract_context_tokens(
            input_filename,
            f"./dynamic_par/azure_trace/b{batch_size}/conv_stdev{batch['stdev']:04d}_{batch['start']:04d}_{batch['end']:04d}.npy",
            batch["start"],
            batch["end"],
            batch_size,
        )
