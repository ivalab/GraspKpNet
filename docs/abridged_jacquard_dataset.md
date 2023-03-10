# README File

**Contact** – Ruinian Xu & rnx94@gatech.edu

**Method(s)** – Based on the original Jacquard Dataset and the ratio of the union of all grasp regions to the masked region of the object, keep those images whose ratio is larger than 0.8, remove those images whose ratio is less than 0.2 and manually check images whose ratio is between 0.2 and 0.8 and remove those images whose grasp annotations satisfy one of four conditions. These conditions can be referred to the manuscript (https://arxiv.org/pdf/2106.08497.pdf).

**Data Structure** – The root folder should be composed of train and test sub-folders. The train folder contains two sub-folders which are annotations and grasps_train2018. The train_annotations_0_5.tar.gz and train_annotations_6_11.tar.gz zip files should be uncompressed into annotations folder. The train_grasps_train2018_0_5.tar.gz and train_grasps_train2018_6_11.tar.gz zip files should be uncompressed into grasps_train2018 folder. Besides these two sub-folders, the instances_grasps_train2018_edge_denseanno_filter.json and instances_grasps_train2018_filter.json should be put into train folder. The test folder contains two sub-folders which are annotations and grasps_test2018. The test_annotations_0_5.tar.gz and test_annotations_6_11.tar.gz zip files should be uncompressed into annotations folder. The train_grasps_test 2018_0_5.tar.gz and train_grasps_test 2018_6_11.tar.gz zip files should be uncompressed into grasp_test2018 folder. Besides these two sub-folders, the instances_grasps_test2018_edge_denseanno_filter.json, instances_grasps_test2018_filter.json and instances_grasps_test2018 \_filter.json should be put into test folder.
File Information – The root folder is composed of train and test folders. For each folder, it has two folders store annotations and grasp images, and two json files store filtered grasp annotations for Lm-RM grasp representation and full grasp annotations for edge Lm-Rm grasp representation. Additionally, test folder contains full grasp annotations for Lm-RM grasp representation.

**Software** – Zip is required to uncompress files.

```
Abridged Jacquard Dataset
|--- train/
|    |--- annotations/
|    |    |--- train_annotations_0_5.tar.gz
|    |__ train_annotations_6_11.tar.gz zip
|    |    |--- grasps_train2018/
|    |    |--- train_grasps_train2018_0_5.tar.gz
|    |    |__ train_grasps_train2018_6_11.tar.gz
|    |--- instances_grasps_train2018_edge_denseanno_filter.json
|    |__ instances_grasps_train2018_filter.json
|--- test/
|	|--- annotations/
|	|   |--- test_annotations_0_5.tar.gz
|	|   |__ test_annotations_6_11.tar.gz
|	|--- grasps_test2018/
|	|   |--- train_grasps_test 2018_0_5.tar.gz
|	|   |__ train_grasps_test 2018_6_11.tar.gz
|	|--- instances_grasps_test2018_edge_denseanno_filter.json
|	|--- instances_grasps_test2018_filter.json
|	|__ instances_grasps_test2018 _filter.json
|__ README.docx
```

**Data source** - Simulation;
A. Depierre, E. Dellandrea and L. Chen, “Jacquard: A Large Scale Dataset for Robotic Grasp Detection,” in IEEE International Conference on Intelligent Robots and Systems, 2018

**Location where data were collected** - IVALab

**Time period during which data were collected** - 16/09/2019-23/09/2019

**Date dataset was last modified** - 23/09/2019

**Dataset identifier** - http://hdl.handle.net/1853/64897

**Affiliated manuscript**: arXiv:2106.08497
