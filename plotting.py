import os

import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def plot_time_series(self, train_data, test_data, file_name="time_series_plot.png"):
        plt.figure(figsize=(10, 6))

        item_ids = train_data.index.get_level_values("item_id").unique()

        for item_id in item_ids:
            train_subset = train_data.xs(item_id, level="item_id")
            test_subset = test_data.xs(item_id, level="item_id")

            plt.plot(
                train_subset.index,
                train_subset["target"],
                label=f"Train Item {item_id}",
            )
            plt.plot(
                test_subset.index,
                test_subset["target"],
                label=f"Test Item {item_id}",
                linestyle="--",
            )

        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.title("Train and Test Time Series Data")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_folder, file_name)
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
