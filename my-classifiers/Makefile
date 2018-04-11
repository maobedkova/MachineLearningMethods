all: download perc nb knn

download:
	wget -O artificial_objects.tgz http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz
	wget -O adult.data.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
	wget -O adult.test.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
	tar -xvzf artificial_objects.tgz
	tail -n +2 "adult.test.csv" > "adult.test.tmp" && mv "adult.test.tmp" "adult.test.csv"

perc:
	python3 perceptron.py
	
nb:
	python3 naive_bayes.py

knn:
	python3 knn.py

