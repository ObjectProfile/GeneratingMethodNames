# Generation of method names

This repository contains all the source code related to the research project of [Oleksandr Zaytsev](@olekscode) performed under the supervision of [Alexandre Bergel](@bergel) at the [University of Chile](http://www.uchile.cl/). We apply the techniques of neural machine translation to generate method names based on source code of [Pharo](https://pharo.org/). In this setting source code becomes the source language (from which we translate) and English, in which the words composing the method names are written, becomes the target language (to which the source code is translated). Therefore we are dealing with the problem of translating source code of Pharo into English. We argue that in a way this problem is equivalent to the translation between human languages, such as English and French.

## Project structure

* **data/** - empty folder where you should put the dataset of Pharo methods (follow the instructions to download or construct this dataset). This folder will also store additional files generated during training, such as train/validation/test pairs and vocabularies.
* **img/** - different visualizations generated during training.
* **models/** - empty folder where you can put the pretrained model (follow the instructions to download it). If you choose to train your own model, it will also be stored in this folder.
* **src/** - source code of this project.

## Reproducing the results

These are the step-by-step instructions for reproducing our experiment.

### Step 1. Get the data

The following steps require the file `methods_tokenized.csv` in `data/` folder. You can either download this file from our Google Drive or construct it yourself from the methods in your Pharo image, following the instructions for collecting the data.

### Step 2. Choose the logging mode

Training takes a lot of time. Every couple of iterations (1/100 of the total number of iterations) the intermediary results are reported and stored into log files. By default these logs are written onto your disk. If you choose to keep it this way, you can skip the rest of this step and jump right to step 3.

Alternatively, you can send the logs to your Google Drive. This way you can train the model on a remote machine and receive live updates, which you can view anywhere, even on your smartphone.

### Step 3. Train the model 

In order to train the model you should go to `src/model/` folder

```bash
cd src/model/
```

Run the following command to start the training
```bash
python model.py --train
```
100,000 iterations of training take around 10 hours on Intel Core i5 CPU. So be patient or get yourself a GPU.