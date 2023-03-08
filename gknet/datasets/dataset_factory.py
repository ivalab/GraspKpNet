from .sample.dbmctdet import DbMCTDetDataset
from .sample.dbmctdet_cornell import DbMCTDet_CornellDataset

from .dataset.jac_coco import JAC_COCO_36
from .dataset.cornell import CORNELL


dataset_factory = {
    "jac_coco_36": JAC_COCO_36,
    "cornell": CORNELL,
}

_sample_factory = {
    "dbmctdet": DbMCTDetDataset,
    "dbmctdet_cornell": DbMCTDet_CornellDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
