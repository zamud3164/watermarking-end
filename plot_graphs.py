import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
runs_folder = "crop_30"
#runs_folder = "dropout_30_v2"
#runs_folder = "gaussian_noise_01_v2"
#runs_folder = "jpeg_50_v2"
#runs_folder = "median_blur_3_v2"

#column_to_plot = "bitwise-error"
column_to_plot = "encoder_mse"

train_file = f"runs/{runs_folder}/train.csv"
val_file = f"runs/{runs_folder}/validation.csv"
plot_dir = f"runs/{runs_folder}/plots_svg"
plot_filename = f"{column_to_plot}_{runs_folder}.svg"
plot_path = os.path.join(plot_dir, plot_filename)

# === Ensure plot directory exists ===
os.makedirs(plot_dir, exist_ok=True)

# === Load the data ===
df_train = pd.read_csv(train_file, sep=",")
df_val = pd.read_csv(val_file, sep=",")

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(df_train['epoch'], df_train[column_to_plot], label='Train', color='blue')
plt.plot(df_val['epoch'], df_val[column_to_plot], label='Validation', color='orange')

plt.title(f"{column_to_plot.capitalize()} for {runs_folder.capitalize()}")
plt.xlabel("Epoch")
plt.ylabel(column_to_plot.capitalize())
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save and show plot ===
plt.savefig(plot_path)
#plt.show()
