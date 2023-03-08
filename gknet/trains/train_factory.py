from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dbmctdet import DbMCtdetTrainer

train_factory = {
  'dbmctdet': DbMCtdetTrainer,
  'dbmctdet_cornell': DbMCtdetTrainer,
}
