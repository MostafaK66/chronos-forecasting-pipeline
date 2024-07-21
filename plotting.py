import os
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def plot_time_series(self, df, file_name='time_series_plot.png'):
        plt.figure(figsize=(10, 6))
        for item_id in df['item_id'].unique():
            subset = df[df['item_id'] == item_id]
            plt.plot(subset['timestamp'], subset['value'], label=f'Item {item_id}')

        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Time Series Data')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_folder, file_name)
        plt.savefig(output_path)
        plt.close()
        print(f'Plot saved to {output_path}')
