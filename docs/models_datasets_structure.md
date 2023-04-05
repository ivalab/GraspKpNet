# models and datasets structure

It's useful to know what files the repository expects in models and datasets.
These can be extended as needed.

## models

Models are written out to the `models` directory.

```
$ tree models
models
|-- backup
|   `-- model_dla34_cornell.pth
|-- ctdet_coco_dla_2x.pth
|-- model_alexnet_ajd.pth
|-- model_alexnet_cornell.pth
|-- model_dla34_ajd.pth
|-- model_dla34_cornell.pth
|-- model_hg104_ajd.pth
|-- model_hg104_cornell.pth
|-- model_hg52_ajd.pth
|-- model_hg52_cornell.pth
|-- model_resnet18_ajd.pth
|-- model_resnet18_cornell.pth
|-- model_resnet50_ajd.pth
|-- model_resnet50_cornell.pth
|-- model_vgg16_ajd.pth
`-- model_vgg16_cornell.pth
```

Note that the copy of the models in the archive has a typo for the `model_dla34_cornell.pth` file.
This should be corrected from `model_dl34_cornell.pth` to `model_dla34_cornell.pth`.

## datasets

This is the top level for the datasets directory.

```
$ tree datasets -L 4
datasets
|-- Cornell
|   `-- rgd_5_5_5_corner_p_full
|       `-- data
|           |-- Annotations
|           |-- ImageSets
|           `-- Images
`-- Jacquard
    `-- coco
        `-- 512_cnt_angle
            |-- test
            `-- train
```

We note that there are a significant number of files under the `Annotations` directory.

```
$ ls datasets/Cornell/rgd_5_5_5_corner_p_full/data/Annotations | wc -l
110625
```

The Jacquard dataset is a bit different, and is split across a `test` and `train` directory.

```
$ tree datasets/Jacquard -L 5
datasets/Jacquard
`-- coco
    `-- 512_cnt_angle
        |-- test
        |   |-- annotations
        |   |   |-- Jacquard_Dataset_0
        |   |   |-- Jacquard_Dataset_1
        |   |   |-- Jacquard_Dataset_10
        |   |   |-- Jacquard_Dataset_11
        |   |   |-- Jacquard_Dataset_2
        |   |   |-- Jacquard_Dataset_3
        |   |   |-- Jacquard_Dataset_4
        |   |   |-- Jacquard_Dataset_5
        |   |   |-- Jacquard_Dataset_6
        |   |   |-- Jacquard_Dataset_7
        |   |   |-- Jacquard_Dataset_8
        |   |   `-- Jacquard_Dataset_9
        |   |-- grasps_test2018
        |   |   |-- Jacquard_Dataset_0
        |   |   |-- Jacquard_Dataset_1
        |   |   |-- Jacquard_Dataset_10
        |   |   |-- Jacquard_Dataset_11
        |   |   |-- Jacquard_Dataset_2
        |   |   |-- Jacquard_Dataset_3
        |   |   |-- Jacquard_Dataset_4
        |   |   |-- Jacquard_Dataset_5
        |   |   |-- Jacquard_Dataset_6
        |   |   |-- Jacquard_Dataset_7
        |   |   |-- Jacquard_Dataset_8
        |   |   |-- Jacquard_Dataset_9
        |   |   `-- train_grasps_test2018_6_11.tar
        |   |-- instances_grasps_test2018.json
        |   |-- instances_grasps_test2018_edge_denseanno_filter.json
        |   `-- instances_grasps_test2018_filter.json
        `-- train
            |-- annotations
            |   |-- Jacquard_Dataset_0
            |   |-- Jacquard_Dataset_1
            |   |-- Jacquard_Dataset_10
            |   |-- Jacquard_Dataset_11
            |   |-- Jacquard_Dataset_2
            |   |-- Jacquard_Dataset_3
            |   |-- Jacquard_Dataset_4
            |   |-- Jacquard_Dataset_5
            |   |-- Jacquard_Dataset_6
            |   |-- Jacquard_Dataset_7
            |   |-- Jacquard_Dataset_8
            |   `-- Jacquard_Dataset_9
            |-- grasps_train2018
            |   |-- Jacquard_Dataset_0
            |   |-- Jacquard_Dataset_1
            |   |-- Jacquard_Dataset_2
            |   |-- Jacquard_Dataset_3
            |   |-- Jacquard_Dataset_4
            |   `-- Jacquard_Dataset_5
            |-- instances_grasps_train2018_edge_denseanno_filter.json
            `-- instances_grasps_train2018_filter.json
```

```
$ ls datasets/Jacquard/coco/512_cnt_angle/test/annotations/Jacquard_Dataset_0/ | head -n10
1a0312faac503f7dc2c1a442b53fa053
1a0710af081df737c50a037462bade42
1a2a5a06ce083786581bb5a25b17bed6
1a30adabf5a2bb848af30108ea9ccb6c
1a3efcaaf8db9957a010c31b9816f48b
1a46011ef7d2230785b479b317175b55
1a477f7b2c1799e1b728e6e715c3f8cf
1a4daa4904bb4a0949684e7f0bb99f9c
1a5327b328cd97d084c3569473be6c23
1a5f561ce4cbca2625c70fb1df3f879b
```

Each dataset in the training set represents roughly 550 images, while each dataset in the test set represents roughly 900 images.
