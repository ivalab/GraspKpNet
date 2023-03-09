from .dbmctdet import DbMCtdetDetector
from .dbmctdet_cornell import DbMCtdetDetector_Cornell

detector_factory = {
    "dbmctdet": DbMCtdetDetector,
    "dbmctdet_cornell": DbMCtdetDetector_Cornell,
}
