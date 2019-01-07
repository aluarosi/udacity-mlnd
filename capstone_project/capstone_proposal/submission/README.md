# Capstone Proposal

Álvaro Santamaría Herrero

December 2018

The capstone project proposal is described in the companion file: `./proposal.pdf`

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

