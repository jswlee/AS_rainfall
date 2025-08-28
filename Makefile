install:
	python -m pip install --upgrade pip setuptools wheel &&\
	pip install -e ".[all]"
tune:
	python -m Hyperparameter_Tuning.pytorch_hyperparameter_tuning
train:
	python -m Train_Best_Model.train_land_model