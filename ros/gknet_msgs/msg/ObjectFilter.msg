# An object containing information about corners to help filter points in a
# given area. This format is inspired by the COCO format.
# https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#cocodetectiondataset

# 2D bounding box of object in the image in pixels, in the format [xtl, ytl, xbr, ybr]
# tl = top left, br = bottom right
uint16[4] bbox
