from load import load
from baselines import model_all_mentions_paired, model_all_mentions_unpaired
from evaluate import get_avg_metrics
from multiword_exact_match import model_pair_exact_strings, multiword_exact_match
from trained_model import LinearEdgePredictor
from huggingface_neural_coref import run_hugging_face
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Perform various coreference resolution techniques.")
    parser.add_argument("--test-filename",
                        dest="test_filename",
                        type=str,
                        default="PreCo_1.0/dev.jsonl",
                        help="Specify the filename for the test data.")
    parser.add_argument("--train-filename",
                        dest="train_filename",
                        type=str,
                        default="PreCo_1.0/train.jsonl",
                        help="Specify the filename for the training data.")
    parser.add_argument("--train-load-limit",
                        dest="train_load_limit",
                        type=int,
                        default=100,
                        help="Specify the number of documents to load for the training data. If the limit is above the "
                             "number of documents, then all documents will be loaded.")
    parser.add_argument("--test-load-limit",
                        dest="test_load_limit",
                        type=int,
                        default=100,
                        help="Specify the number of documents to load for the test data. If the limit is above the "
                             "number of documents, then all documents will be loaded.")
    parser.add_argument("--all-mentions-unpaired", "-u",
                        dest="all_mentions_unpaired",
                        action="store_true",
                        help="If this flag is active, then the program will only run the unpaired baseline and other "
                             "explicitly specified coreference methods. The unpaired baseline places each coreference"
                             "in its own cluster.")
    parser.add_argument("--all-mentions-paired", "-p",
                        dest="all_mentions_paired",
                        action="store_true",
                        help="If this flag is active, then the program will only run the paired baseline and other "
                             "explicitly specified coreference methods. The paired baseline places every coreference"
                             "in the same cluster. ")
    parser.add_argument("--multiword-exact-match", "-m",
                        dest="multiword_exact_match",
                        action="store_true",
                        help="If this flag is active, then the program will only run the multiword exact match method"
                             " and other explicitly specified coreference methods. The multiword exact match method "
                             "places every coreference in the same cluster as other coreferences with the exact same "
                             "string representation.")
    parser.add_argument("--linear-edge-predictor", "-l",
                        dest="linear_edge_predictor",
                        action="store_true",
                        help="If this flag is active, then the program will only run the linear edge predictor method"
                             " and other explicitly specified coreference methods. The linear edge predictor method "
                             "trains a linear regression model to predict whether or not a pair of mentions should have"
                             "an edge connecting them, and combines all paired mentions into clusters.")
    parser.add_argument("--hugging-face", "-f",
                        dest="hugging_face",
                        action="store_true",
                        help="If this flag is active, then the program will only run the hugging face method"
                             " and other explicitly specified coreference methods. The hugging face method "
                             "uses a feed-forward neural network to identify mentions and combines all paired mentions "
                             "into clusters.")
    parser.add_argument("--dataset-stats", "-d",
                        dest="dataset_stats",
                        action="store_true",
                        help="If this flag is active, then the program will output statistics about the dataset.")
    parser.add_argument("--quiet", "-q",
                        dest="quiet",
                        action="store_true",
                        help="Display fewer progress bars and extraneous information. Program may appear to hang.")
    parser.add_argument("--dry-run", "-n",
                        dest="dry_run",
                        action="store_true",
                        help="If this flag is active, then no actions will be performed other than loading the data to"
                             "verify its integrity. Dataset statistics may still be output through -d or "
                             "--dataset-stats, but no coreference resolution will be done regardless of other flags.")
    return parser.parse_args()

def main():
    args = get_arguments()
    test_filename = args.test_filename
    train_filename = args.train_filename

    train_data = load(train_filename, limit=args.train_load_limit)
    test_data = load(test_filename, limit=args.test_load_limit)
    if args.dataset_stats:
        print("Training Data Documents: {}".format(len(train_data)))
        print("Test Data Documents: {}".format(len(test_data)))
        print("Training Mentions: {}".format(sum(len(doc.mentions) for doc in train_data)))
        print("Test Mentions: {}".format(sum(len(doc.mentions) for doc in test_data)))
    specified = args.all_mentions_unpaired or args.all_mentions_paired or args.multiword_exact_match or \
        args.linear_edge_predictor or args.hugging_face
    verbose = not args.quiet
    if args.dry_run:
        return 0
    if not specified:
        precision, recall, f1 = get_avg_metrics(test_data, model_all_mentions_unpaired, {}, verbose=verbose)
        print("--------ALL MENTIONS UNPAIRED--------")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        precision, recall, f1 = get_avg_metrics(test_data, model_all_mentions_paired, {}, verbose=verbose)
        print("---------ALL MENTIONS PAIRED---------")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        precision, recall, f1 = get_avg_metrics(test_data, model_pair_exact_strings,
                                                {'match': multiword_exact_match}, verbose=verbose)
        print("--------MULTIWORD EXACT MATCH--------")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        edge_predictor = LinearEdgePredictor(train_data, verbose=verbose)
        precision, recall, f1 = get_avg_metrics(test_data, edge_predictor.get_predictions, {}, verbose=verbose)
        print("--------LINEAR EDGE PREDICTOR--------")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        precision, recall, f1 = get_avg_metrics(test_data, run_hugging_face, {}, verbose=verbose)
        print("------------HUGGING FACE-------------")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
    else:
        if args.all_mentions_unpaired:
            precision, recall, f1 = get_avg_metrics(test_data, model_all_mentions_unpaired, {}, verbose=verbose)
            print("--------ALL MENTIONS UNPAIRED--------")
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
        if args.all_mentions_paired:
            precision, recall, f1 = get_avg_metrics(test_data, model_all_mentions_paired, {}, verbose=verbose)
            print("---------ALL MENTIONS PAIRED---------")
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
        if args.multiword_exact_match:
            precision, recall, f1 = get_avg_metrics(test_data, model_pair_exact_strings,
                                                    {'match': multiword_exact_match}, verbose=verbose)
            print("--------MULTIWORD EXACT MATCH--------")
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
        if args.linear_edge_predictor:
            edge_predictor = LinearEdgePredictor(train_data, verbose=verbose)
            precision, recall, f1 = get_avg_metrics(test_data, edge_predictor.get_predictions, config={},
                                                    verbose=verbose)
            print("--------LINEAR EDGE PREDICTOR--------")
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
        if args.hugging_face:
            precision, recall, f1 = get_avg_metrics(test_data, run_hugging_face, {}, verbose=verbose)
            print("------------HUGGING FACE-------------")
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))


if __name__ == "__main__":
    main()