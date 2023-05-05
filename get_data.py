mport os
import wget

# data from https://www.kaggle.com/datasets/valentynsichkar/mnist-preprocessed/download?datasetVersionNumber=3
# Download the zipped dataset
url = 'www.kaggle.com/datasets/valentynsichkar/mnist-preprocessed'
file_name = "data.csv"
wget.download(url, file_name)
