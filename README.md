# channel_image_segmentation

This is a collabrative repositary hosting code to identify channel pixels from a flume experiment image, with optionally additional data like elevation maps. 

## data

Example data sets are provided in `data/test_data`. The images in the folder are in groups of three: one correspond to the mask (target for the model), one correspond to the RGB image, and another one correspond to the elevation model (optional data). For example:

![mask](data/test_data/part_0_batch_0_num_0_mask.png)
![image](data/test_data/part_0_batch_0_num_0_image.png)
![elevation](data/test_data/part_0_batch_0_num_0_elevation.png)

The module ``data_preparing`` provides methods to generate the groups of images from a data source like ``numpy.ndarray`` or matlab's ``.mat`` files.

## requirements

We recommend using virtual environments to isolate the program environment. The file `requirements.txt` contains a list of module needed for the project. To setup an environment using ``python3-venv``:

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/active
pip3 install -r requirements.txt
```

This project uses [Tensorflow](https://www.tensorflow.org/) to construct the CNN models. 

GPU or better resource is recommended to train the model. One could experiment or demo with a small sample dataset on a CPU only environment.
