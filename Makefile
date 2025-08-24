install:
	python3 -m pip install --upgrade pip setuptools wheel &&\
	pip install -e ".[all]"
tune:
	python3 -m Hyperparameter_Tuning.pytorch_hyperparameter_tuning
train:
	python3 -m Train_Best_Model.pytorch_train_best_model