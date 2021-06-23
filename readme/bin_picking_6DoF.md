### Clutter Clearance or Bin-Picking Outcomes for 6-DOF Grasping Nethods

| Method | #Total | #Placed | #Trials | Results | SR  | PC  | TC |
| :----- | :----: | :-----: | :-----: | :------ | :-: | :-: | :-: |
| CollisionNet [1] | 51 | 4-9 | 9 | 41/51 | 80.4% | N/A | 1T |
| Contact-GraspNet [2] | 51 | 4-9 | 9 | 53/59 | 90.2% | N/A | 2T |
|                     |    |     |   |       | 84.3% (first attempt) | N/A | 1T |
| GPD [3] | 27 | 10 | 30 | 266/288 | 93.0% | 89% | 3F |
| [4] | 30 | 30 | 10 | 300/399 | 75.2% | 100% | 10F |
| [5] | 8 | 8 | 5 | 40/45 (S1) | 89.3% | 100% | N/A |
|     | 8 | 8 | 5 | 38/57 (S2) | 66.2% | 95% | N/A |
| [6] | 15 | 6 | 20 | 117/141 (known objects) | 83.0% | 97.5% | 10A |
|     | 15 | 6 | 20 | 110/154 (novel objects) | 71.4% | 91.6% | 10A |
| [7] | N/A | N/A | 10 | N/A | 79.3% | 96.0% | 15A |
| [8] | 30 | 10 | 4 | 37/48 | 77.1% | 92.5% | N/A |
| [9] | 10 | 10 | 10 | 113/155 | 72.9% | 85.0% | 3F |

Legend: Termination Condition (TC) \
*x*T - _x_ try per object. \
*x*F - Terminate when _x_ consecutive grasp failures occur for the same object or all objects removed.\
*x*A - Max number of attempts for each run is limited to _x_ or all objects removed.\

- [1] 6-DOF Grasping for Target-driven Object Manipulation in Clutter
- [2] Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes
- [3] Grasp Pose Detection in Point Clouds
- [4] Learning to Generate 6-DoF Grasp Poses with Reachability Awareness
- [5] PointNetGPD: Detecting Grasp Configurations from Point Sets
- [6] PointNet++ Grasping: Learning An End-to-end Spatial Grasp Generation Algorithm from Sparse Point Clouds
- [7] REGNet: REgion-based Grasp Network for End-to-end Grasp Detection in Point Clouds
- [8] S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes
- [9] Using Geometry to Detect Grasp Grasps in 3D Point Clouds
