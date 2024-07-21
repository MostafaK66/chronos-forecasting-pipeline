import os
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def plot_time_series(self, df, file_name='time_series_plot.png'):
        plt.figure(figsize=(10, 6))

        item_ids = df.index.get_level_values('item_id').unique()

        for item_id in item_ids:
            subset = df.xs(item_id, level='item_id')
            plt.plot(subset.index, subset['y'], label=f'Item {item_id}')

        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Time Series Data')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_folder, file_name)
        plt.savefig(output_path)
        plt.close()
        print(f'Plot saved to {output_path}')

