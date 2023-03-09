from .dbmctdet import DbMCtdetTrainer

train_factory = {
    "dbmctdet": DbMCtdetTrainer,
    "dbmctdet_cornell": DbMCtdetTrainer,
}
