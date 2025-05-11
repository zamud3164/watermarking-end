### Codebase for Thesis: Robust Image Watermarking Using Encoder-Decoder Networks with Feature-Guided Embedding and Adaptive Strength Control

This project presents a deep learning-based image watermarking system built on an END (Encoder–Noise Layer–Decoder) architecture. It introduces architectural improvements including a feature extractor, adaptive strength factor adjustment, and Squeeze-and-Excitation (SE) blocks to improve robustness and imperceptibility. The system was evaluated against state-of-the-art watermarking models and includes a user study to assess perceptual quality.

The data is not given in this repository. It was too big and therefore it was not possible to upload it to the git repository. The data can be found in Google Drive. The link to the folder is given in thesis, in Section 6.1. Below follows instructions on how to run this project in Google Colab. The whole project, including data, is given in the Google Drive folder.

To run this project, run the HiDDeN.ipynb file. This is a Python notebook and the 4 training stages described in the thesis report is defined in this notebook. First, mount the google drive and install a necessary package. Afterwards, run the four stages. Lastly, test the model. The link to the google drive folder is given in the thesis report, and all of the necessary files including the dataset is uploaded in the given folder to make it easy to run it in Google Colab. It is possible to run the project locally to, download the dataset from the drive folder and place it locally, and then change the --datadir parameters in the notebook to the actual paths to the dataset when run locally. If run in Google Colab, no changes are needed to be made. 

Python and PyTorch is needed to run this watermarking system and train models. 

## Command Reference
Below are the four command for the four stages in this training. IT includes a breakdown of the different parameters used in the commands. 

### Stage 1 - Start the training of a model
<pre>!python drive/MyDrive/HiDDeN/main.py new --name new_model_exam --epochs 30 --data-dir drive/MyDrive/HiDDeN/data --batch-size 16 --noise 'medianblur(3)'</pre>

**Parameter Breakdown:**

- `--name`: The name for the model/run. This creates a subfolder under `runs/` where all logs and checkpoints are stored.
- `--epochs`: Number of training epochs to run.
- `--data-dir`: Directory containing the training dataset.
- `--batch-size`: Number of images per training batch.
- `--noise`: Specifies a noise transformation to apply during training (for example: 'medianblur(3)').

---

### Stage 2 – Continue Training the Model
<pre>!python drive/MyDrive/HiDDeN/main.py continue --folder drive/MyDrive/HiDDeN/runs/new_model_exam --epochs 50</pre>

**Parameter Breakdown:**

- `--folder`: Path to the folder where previous model checkpoints and logs are stored.
- `--epochs`: Number of additional training epochs to run.

---

### Stage 3 – Fine-tune the Model with Adaptor
<pre>!python drive/MyDrive/HiDDeN/main.py finetune --folder /content/drive/MyDrive/HiDDeN/runs/new_model_exam --epochs 100 --use-adaptor</pre>

**Parameter Breakdown:**

- `--folder`: Path to the folder containing previous model checkpoints.
- `--epochs`: Number of fine-tuning epochs to run.
- `--use-adaptor`: Flag to enable the use of the adaptor network for fine-tuning.

---

### Stage 4 – Fine-tune the Model with Stage-4 and Adaptor
<pre>!python drive/MyDrive/HiDDeN/main.py finetune --folder /content/drive/MyDrive/HiDDeN/runs/new_model_exam --epochs 200 --use-adaptor --stage stage-4</pre>


**Parameter Breakdown:**

- `--folder`: Path to the folder containing previous model checkpoints.
- `--epochs`: Number of fine-tuning epochs to run.
- `--use-adaptor`: Flag to enable the use of the adaptor network for fine-tuning.
- `--stage`: Specifies the training stage, used only for stage 4 so the "finetune" command can be used but with different hyperparameters than in stage 3.


## Codebase Structue
The codebase has four folders: data, models, runs, noise_layers. The data folder includes the training, validation and test data folders. They are given as paths in the commands when running the training stages. The next folder, models, includes all python files for building the model. It has the encoder, decoder, feature extractor, adaptor, and discriminator architecture files. noise_layers has all the files for applying different distortions to the images. runs has the checkpoints for the different models that are run. The rest of the necessary files to run the system is in the root folder. The HiDDeN notebook, which runs all the training commands as well as the model testing is also in the root folder. 

