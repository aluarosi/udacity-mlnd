# Capstone Project

Udacity Machine Learning Engineer Nanodegree. Capstone Project.

Title: **Multi-Class Text Classification wiht Convolutional Neural Nets**
 
Author: **[Álvaro Santamaría Herrero](https://github.com/aluarosi)**
 
January 2019. 

GitHub: [https://github.com/aluarosi/udacity-mlnd/capstone_project](https://github.com/aluarosi/udacity-mlnd)


## Deliverable files

These are the deliverable files:

* Project proposal: `project_proposal.pdf`
* Project report: `project_report.pdf`
* Project notebook (ipython): `project_notebook.ipynb`
* Python code (converted from the notebook): `project_notebook.py`
* Support files (training reports and weights of trained models) in directory `./wrk/`. See more information below.

Not delivered for evaluation but included in the git repository:

* Source of project proposal document in *markdown* format: `./capstone_proposal/capstone_proposal.md`
* Sources of project report in *LaTeX*: `./capstone_report/overleaf/`
* Images: `./images/`

## Supporting material

### The 20 newsgroups text dataset 

The dataset **"20 newsgroups"** is obtained using the utility function `sklearn.datasets. fetch_20newsgroups` of the library scikit-learn. See [sklearn documentation](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) and the following example referred there:

```
>>> from sklearn.datasets import fetch_20newsgroups
>>> newsgroups_train = fetch_20newsgroups(subset='train')

>>> from pprint import pprint
>>> pprint(list(newsgroups_train.target_names))
['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
```

### Glove embeddings 

The GloVe embeddings (GloVe: Global Vectors for Word Representation) can be downloaded from the [GloVe web page](https://nlp.stanford.edu/projects/glove/). We've downloaded the file [glove6B.zip](http://nlp.stanford.edu/data/glove.6B.zip) which in the GloVe page is annotated with: 

> Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download).

From `glove.6B.zip` file we've used the file `glove.6B.100d.txt`, which correspond to vectors of 100 dimensions.

### Training reports and pre-trained model weights

As mentioned, we include in  `./wrk/` a set of files with:

* Reports from hyperparameter exploration for the different models. 
* Weights of the trained models. 
* Reports of the variance analysis (which encompasses repeated traning and evaluation of the models). 

```
VARIANCE_ANALYSIS_20181229144242
VARIANCE_ANALYSIS_20181230120812
modelA_report.pickle
modelA_weights.pdf5
modelB_report.pickle
modelB_weights.pdf5
modelC_report.pickle
modelC_weights.pdf5
reports0_20181201183104.pickle
reports0_with_repetitions_20181226084937.pickle
reports1_with_repetitions_20181226165557.pickle
reports2_with_repetitions_A_00.pickle
reports2_with_repetitions_A_01.pickle
reports2_with_repetitions_A_02.pickle
reports2_with_repetitions_A_03.pickle
reports2_with_repetitions_A_04.pickle
reports2_with_repetitions_A_05.pickle
reports2_with_repetitions_A_06.pickle
reports2_with_repetitions_A_20181228.pickle
reportsD_20181228174155.pickle
```

When you execute the notebook you may choose to go through the hyperparamter exploration, training and evaluation processes, or skip some cells and load our intermediate and final results from the files.

## Code dependencies

The code is written in **Python 3** and we used the following libraries:

* [NumPy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://www.scipy.org/)
* [Keras](https://keras.io/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)

Version information:

```
Python version: 3.6.7
Libraries:
    keras: 2.2.4
    matplotlib: 2.1.2
    numpy: 1.14.6
    pandas: 0.22.0
    scipy: 1.1.0
    seaborn: 0.9.0
    sklearn: 0.20.2
```

## Additional setup

1. Download the mentioned embedings file `glove.6B.100d.txt` and copy it in the `./wrk/` directory.
2. Training neural networks is computationally expensive. We recommend to run the notebook `project_notebook.ipynb` in an environment with **GPU support**, e.g., Google Colab.
3. Open the notebook `project_notebook.ipynb` either with Jupyter: `$ jupyter notebook ./project_notebook.ipynb` or in Google Colab.
4. When working in Google Colab, the notebook will connect to an associated Google Drive account, where you should copy the `./wrk/` directory.
5. Configure manually the variables `ENV` and `BASE_DIR` (local vs. Google Colab environment, and path base directory`). Then, run the notebook.

```
#
# Two environments: Google Colab and local (Spyder).
# Set ENV reference.
#
ENV = 'colab'
#ENV = 'local'

#
# Set manually your working directories
#
BASE_DIR = '/content/drive/My Drive/UDACITY MLND' \
            if ENV == 'colab' else\
            '/Volumes/PENDRIVENEW/live!/PROJECTS/UDACITY_ML_NANODEGREE/my_MLND_projects/nlp-cnn'
```

## Online tools

We resorted the following online tools:

* [Google Colab](https://colab.research.google.com). GPU-enabled python notebook environment.
* [Overleaf](https://es.overleaf.com/). Online LaTeX editor.
