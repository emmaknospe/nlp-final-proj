from display import display_coreferences
from load import load
from baselines import model_all_mentions_paired, model_all_mentions_unpaired
from evaluate import get_avg_metrics
from multiword_exact_match import model_pair_exact_strings, multiword_exact_match
from trained_model import LinearEdgePredictor


def main():
    # arg parsing eventually here
    dev_filename = "PreCo_1.0/dev.jsonl"
    train_filename = "PreCo_1.0/train.jsonl"
    mention_method = "extract"

    train_data = load(train_filename, mention_method=mention_method, limit=100)
    edge_predictor = LinearEdgePredictor(train_data)
    dev_data = load(dev_filename, mention_method=mention_method, limit=100)
    print(get_avg_metrics(dev_data, model_pair_exact_strings, {'match': multiword_exact_match}))
    print(get_avg_metrics(dev_data, model_all_mentions_unpaired, {}))
    print(get_avg_metrics(dev_data, model_all_mentions_paired, {}))
    print(get_avg_metrics(dev_data, edge_predictor.get_predictions, {}))

if __name__ == "__main__":
    main()