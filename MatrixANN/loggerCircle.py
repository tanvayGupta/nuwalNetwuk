#This script is used for logging the data, which I will use in my 'expert' analysis down the road. 
# My Current Model will be the baseline on which I will build RMSProp and ADAM yay

import csv
import os

OUTPUT_FILE = "CircleBoundary_GD_base.csv"


def save_results(mean_absolute_error, iterations, learning_rate, hidden_neurons):
    """
    Append the results of one training run to Output FIle.

    mae:
        mean absolute error

    iterations:
        Number of training iterations.

    learning_rate:
        Learning rate used for training.
    """

    # if len(errors) != 8:
        # raise ValueError("errors must contain exactly 8 test errors.")

    # mean_absolute_error = sum(abs(error) for error in errors) / len(errors)

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only when creating the file
        if not file_exists:
            writer.writerow([
                "mean_absolute_error",
                "iterations",
                "learning_rate",
                "hidden_neurons"
            ])

        # Append one row for this training run
        writer.writerow([
            mean_absolute_error,
            iterations,
            learning_rate,
            hidden_neurons
        ])

