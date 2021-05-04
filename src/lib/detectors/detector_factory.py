from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dbmctdet import DbMCtdetDetector
from .dbmctdet_cornell import DbMCtdetDetector_Cornell

detector_factory = {
  'dbmctdet': DbMCtdetDetector,
  'dbmctdet_cornell': DbMCtdetDetector_Cornell,
}
