all: download train

download:
	wget -O artificial_2x_train.tsv http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/regression_data/artificial_2x_train.tsv
	wget -O artificial_2x_test.tsv http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/regression_data/artificial_2x_test.tsv
	wget -O pragueestateprices_train.tsv http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/regression_data/pragueestateprices_train.tsv
	wget -O pragueestateprices_test.tsv http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/regression_data/pragueestateprices_test.tsv

train:
	python3 regression.py