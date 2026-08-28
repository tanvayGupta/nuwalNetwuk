#This script is used for logging the data, which I will use in my 'expert' analysis down the road. 
# My Current Model will be the baseline on which I will build RMSProp and ADAM yay

import csv
import os

OUTPUT_FILE = "GD_RandnInit.csv"


def save_results(errors, iterations, learning_rate):
    """
    Append the results of one training run to outputs.csv.

    errors:
        List containing the 4 XOR errors in this order:
        [00, 01, 10, 11]

    iterations:
        Number of training iterations.

    learning_rate:
        Learning rate used for training.
    """

    if len(errors) != 4:
        raise ValueError("errors must contain exactly 4 XOR errors.")

    mean_absolute_error = sum(abs(error) for error in errors) / len(errors)

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only when creating the file
        if not file_exists:
            writer.writerow([
                "error_00",
                "error_01",
                "error_10",
                "error_11",
                "mean_absolute_error",
                "iterations",
                "learning_rate"
            ])

        # Append one row for this training run
        writer.writerow([
            errors[0],
            errors[1],
            errors[2],
            errors[3],
            mean_absolute_error,
            iterations,
            learning_rate
        ])

