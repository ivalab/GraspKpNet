We argue in the manuscript that removing redundant keypoints is important for grasp keypoint detection. In this Readme file, we will provide some extra
examples with visualization to help readers further understand its importance.


Let's take the representation of Top-left, Bottom-left and Bottom-right (TlBlBr) for example. First of all, no matter what kind of keypoint representation,
how to group these separate keypoints into the final prediction is an essential problem. The proposed method replies on the predictions of embedding values
for each keypoint, which is the first and important step in the keypoint grouping process. Failing to predict similar
embedding values for keypoints belong to the same grasp instance will significantly affect the following steps.


Secondly, even after correctly grouping different keypoints into the grasp bounding box, there are still more problems for representations with more than 
three keypoints. Due to no geometric constraints during the training process, there is no guarantee that 3 predicted keypoints can form a valid rectangle.
The visualization of two cases are provided in the following Figure. Keypoints A, B and C are ground-truth, while A', B' and C' are predictions. As shown in 
the figure, since there is no geometric constraints that force 3 keypoints to form a rectangle during the detection process, the post-process is required to transform 
3 predicted keypoints into a valid grasping bounding box. As shown in the first case, when keypoints A and B are predicted accurately, as long as keypoint C is predicted 
slightly far away from the ground-truth, the area of the transformed grasp bounding box will be significantly effected. As shown in the second case, 
when keypoint A is neither predicted perfectly, the prediction error will be further aggravated.

![image](https://user-images.githubusercontent.com/27162640/121444414-7a8af280-c95d-11eb-8f52-f41d32ba9441.png)
