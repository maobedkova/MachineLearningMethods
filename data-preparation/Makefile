all: download prepare split visualize show

download:
	wget -O tweets_dataset.zip  http://www.win.tue.nl/~mpechen/projects/smm/CINLP_datasets.zip
	unzip tweets_dataset.zip

get_pacakges:
	pip3 install --user -U -r requirements.txt 
	python3 download_nltk.py
	
prepare:
	python3 data_process.py

split:
	python3 data_split.py
	
visualize:
	python3 plot.py

show:
	display classes.png
	display link_retweet.png
	display wordcloud.png

