
## Installing dependencies
This project is using [poetry](https://python-poetry.org/) as dependency management. Therefore, installing the required dependencies is as easy as this:
```shell
conda create --name smaat-unet python=3.9
conda activate smaat-unet
poetry install
```

In any case a [requirements.txt](requirements.txt) is also added from the poetry export (`poetry export --without-hashes --output requirements.txt`).

Basically, only the following requirements are needed:
```
tqdm
torch
lightning
tensorboard
torchsummary
h5py
numpy
```

---

### Training

For training on the precipitation task we used the [train_precip_lightning.py](train_precip.py) file.
The training will place a checkpoint file for every model in the `default_save_path` `lightning/precip_regression`. After finishing training place the best models (probably the ones with the lowest validation loss) that you want to compare in another folder in `checkpoints/comparison`.
The [test_precip_lightning.py](test_precip.py) will use all models in that folder and calculate the test-losses for the models.
To calculate the other metrics such as Precision, Recall, Accuracy, F1, CSI, FAR, HSS use the script [calc_metrics_test_set.py](calc_metrics_test_set.py).

### Plots
Example code for creating similar plots as in the paper can be found in [plot_examples.ipynb](plot_examples.ipynb).

### Precipitation dataset
The dataset consists of precipitation maps in 5-minute intervals from 2016-2019 resulting in about 420,000 images.

The dataset is based on radar precipitation maps from the [The Royal Netherlands Meteorological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about).
The original images were cropped as can be seen in the example below:
![Precip cutout](Precipitation%20map%20Cutout.png)

If you are interested in the dataset that we used please write an e-mail to: k.trebing@alumni.maastrichtuniversity.nl and s.mehrkanoon@uu.nl

The 50% dataset has 4GB in size and the 20% dataset has 16.5GB in size. Use the [create_dataset.py](create_datasets.py) to create the two datasets used from the original dataset.

The dataset is already normalized using a [Min-Max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).
In order to revert this you need to multiply the images by 47.83; this results in the images showing the mm/5min.

