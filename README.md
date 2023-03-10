# GNN based Multi Object Detection and Tracking

This the official implementation of our paper GNN-based Object Detection and Tracking for Visual Perception [Anas Hamid Ali], [Hemangkumar Parmar], [Kavish Narula]] and
[Dr. Akilan Thangarajah]. December, 2022.

<p>
   <img align = "center" src = "https://github.com/HemangParmar17/GNN-based-Object-Detection-and-Tracking-for-Visual-Perception/blob/main/Visualization/MOT17Segmentation.gif"/> </p>


1. Clone and enter the repository

2. Create an [Anaconda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
   ```
    1. conda env create -f environment.yaml
    2. conda activate GraphSegment
    3. pip install -e tracking_wo_bnw
    4. pip install -e .
   ```
    
3. (**OPTIONAL**) Modify the variables `DATA_PATH`, and `OUTPUT_PATH` in  `src/GraphSegment/path_cfg.py` so that they are set to
your preferred locations for storing datasets and output results, respectively. By default, these paths will be in this project's root under folders
named data and output, respectively.

4. Download the [MOTChallenge data] from (https://motchallenge.net/).

5. Download the reid network, [Tracktor](https://arxiv.org/abs/1903.05625)'s object detector, and trained models:
    ```
    bash main/setup/download_models.sh
    ```
You can configure training and evaluation experiments by modifying the options in `configs/config.yaml`. As for preprocessing, all available options can be found in `configs/preprocessing_cfg.yaml`.    

 ## Running Experiments

For every training/evaluation experiment you can specify a run_id string. This, together with the execution
date will be used to create an identifier for the experiment being run. A folder named after this identifier, containing
model checkpoints, logs and output files will be created  at `$OUTPUT_PATH/experiments`(`OUTPUT_PATH` is specified at`src/GraphSegment/path_cfg.py).`

## Preprocessing

Run Pre-Proccessing on MOT17:
```
python main/preprocess_detects.py
```
All these scripts will store the preprocessed detections in the right locations within $DATA_PATH.

## Training
You can train a model by running:
```
python main/train.py 
```
The reid network was trained with [torchreid](https://github.com/KaiyangZhou/deep-person-reid), by using ResNet50's
default configuration with images resized to 128 x 56, adding two fully connected layers (see `resnet50_fc256` in src/mot_neural_solver/models/resnet.py)
and training for 232 epochs. 

## Evaluation
You can evaluate a trained model on a set of sequences by running:
```
python main/test.py 
```

The weights used and sequences tested are determined by parameters `ckpt_path` and `data_splits.test`, respectively. By default, the weights from the model we provide will be used and the `MOT17` test sequences will be evaluated. The resulting output files yield the following `MOT17 segmenation` metrics on the train/test set:

|    MOT17       | MOTA         | IDF1       |
|  :---:    | :---:        |     :---:      |   
| **Train** |     76.4     |     73.8       |    
| **Test**  |     73.6     |     68.7       |    

## Cross-Validation
As explained in the paper, we perform cross-validation to report the metrics of ablation experiments.
To do so, we divide `MOT17` sequences in 3 sets of train/val splits. For every configuration, we then run
3 trainings, one per validation split, and report the overall metrics.

You can train and evaluate models in this manner by running:
```
RUN_ID=your_config_name
python main/train.py with run_id=$RUN_ID cross_val_split=1
python main/train.py with run_id=$RUN_ID cross_val_split=2
python main/train.py with run_id=$RUN_ID cross_val_split=3
python main/cross_validation.py with run_id=$RUN_ID
```

## Visualization

For the visual results by running:

```
python Visualization/Visualization.py
```

## References
 ```
   @InProceedings{braso_2020_CVPR,
    author={Guillem Bras?? and Laura Leal-Taix??},
    title={Learning a Neural Solver for Multiple Object Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
 
 ```
Please, also consider citing Tracktor if you use it for preprocessing detections:
```
  @InProceedings{tracktor_2019_ICCV,
  author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}, Laura},
  title = {Tracking Without Bells and Whistles},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}}
```
