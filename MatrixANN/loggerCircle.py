#This script is used for logging the data, which I will use in my 'expert' analysis down the road. 
# My Current Model will be the baseline on which I will build RMSProp and ADAM yay

import csv
import os

OUTPUT_FILE = "CircleBoundary_GD_base.csv"


def save_results(errors, iterations, learning_rate, hidden_neurons):
    """
    Append the results of one training run to Output FIle.

    errors:
        Now this has 8 test inputs, for the circle one
        The final four are border cases, yum

    iterations:
        Number of training iterations.

    learning_rate:
        Learning rate used for training.
    """

    if len(errors) != 8:
        raise ValueError("errors must contain exactly 8 test errors.")

    mean_absolute_error = sum(abs(error) for error in errors) / len(errors)

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only when creating the file
        if not file_exists:
            writer.writerow([
                "error_00",
                "error_01",
                "error_02",
                "error_03",
                "error_04",
                "error_05",
                "error_06",
                "error_07",
                "mean_absolute_error",
                "iterations",
                "learning_rate",
                "hidden_neurons"
            ])

        # Append one row for this training run
        writer.writerow([
            errors[0],
            errors[1],
            errors[2],
            errors[3],
            errors[4],
            errors[5],
            errors[6],
            errors[7],
            mean_absolute_error,
            iterations,
            learning_rate,
            hidden_neurons
        ])

