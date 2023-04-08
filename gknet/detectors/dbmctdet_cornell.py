import numpy as np

from gknet.utils.post_process import dbmctdet_cornell_post_process

from .dbmctdet import DbMCtdetDetector


class DbMCtdetDetector_Cornell(DbMCtdetDetector):
    # apply transformation
    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()

        dets = dets.reshape(1, -1, dets.shape[2])

        dets = dbmctdet_cornell_post_process(
            dets.copy(),
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            scale,
            self.opt.num_classes,
            self.opt.ori_threshold,
        )

        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)

        return dets[0]
