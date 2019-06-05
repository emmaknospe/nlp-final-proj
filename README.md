# Final Project for Natural Language Processing
## Data downloads and library installations
In order to use our coreference resolution models, the PreCo dataset must be 
downloaded from https://preschool-lab.github.io/PreCo/. Several python libraries
may also need to be installed, including tqdm, neuralcoref, spacy, and sklearn.

## Repo structure
Our baseline models can be found in baselines.py. 
Our better-than-baseline multi-word exact match model can be found in 
multiword_exact_match.py.
Our better-than-baseline linear model can be found in trained_model.py.
Our implementation of HuggingFace's external API on our data can be found in
huggingface_neural_coref.py.
Our evaluation metrics can be found in evaluate.py.
Other files include methods to load from files and display coreference results,
as well as some constants.

## Command-line arguments
main.py can be run with many different command-line arguments in order to run 
the different coreference resolution models on the indicated data files with 
customized behaviors:

usage: main.py [-h] [--test-filename TEST_FILENAME]
               [--train-filename TRAIN_FILENAME]
               [--train-load-limit TRAIN_LOAD_LIMIT]
               [--test-load-limit TEST_LOAD_LIMIT] [--all-mentions-unpaired]
               [--all-mentions-paired] [--multiword-exact-match]
               [--linear-edge-predictor] [--hugging-face] [--dataset-stats]
               [--quiet] [--dry-run]

Perform various coreference resolution techniques.

optional arguments:
  -h, --help            show this help message and exit
  --test-filename TEST_FILENAME
                        Specify the filename for the test data.
  --train-filename TRAIN_FILENAME
                        Specify the filename for the training data.
  --train-load-limit TRAIN_LOAD_LIMIT
                        Specify the number of documents to load for the
                        training data. If the limit is above the number of
                        documents, then all documents will be loaded.
  --test-load-limit TEST_LOAD_LIMIT
                        Specify the number of documents to load for the test
                        data. If the limit is above the number of documents,
                        then all documents will be loaded.
  --all-mentions-unpaired, -u
                        If this flag is active, then the program will only run
                        the unpaired baseline and other explicitly specified
                        coreference methods. The unpaired baseline places each
                        coreferencein its own cluster.
  --all-mentions-paired, -p
                        If this flag is active, then the program will only run
                        the paired baseline and other explicitly specified
                        coreference methods. The paired baseline places every
                        coreferencein the same cluster.
  --multiword-exact-match, -m
                        If this flag is active, then the program will only run
                        the multiword exact match method and other explicitly
                        specified coreference methods. The multiword exact
                        match method places every coreference in the same
                        cluster as other coreferences with the exact same
                        string representation.
  --linear-edge-predictor, -l
                        If this flag is active, then the program will only run
                        the linear edge predictor method and other explicitly
                        specified coreference methods. The linear edge
                        predictor method trains a linear regression model to
                        predict whether or not a pair of mentions should
                        havean edge connecting them, and combines all paired
                        mentions into clusters.
  --hugging-face, -f    If this flag is active, then the program will only run
                        the hugging face method and other explicitly specified
                        coreference methods. The hugging face method uses a
                        feed-forward neural network to identify mentions and
                        combines all paired mentions into clusters.
  --dataset-stats, -d   If this flag is active, then the program will output
                        statistics about the dataset.
  --quiet, -q           Display fewer progress bars and extraneous
                        information. Program may appear to hang.
  --dry-run, -n         If this flag is active, then no actions will be
                        performed other than loading the data to verify its
                        integrity. Dataset statistics may still be output
                        through -d or --dataset-stats, but no coreference
                        resolution will be done regardless of other flags.
