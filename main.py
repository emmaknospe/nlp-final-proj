from display import display_coreferences
from load import load
from baseline import model_baseline
from evaluate import evaluate_b3

def main():
    # arg parsing eventually here
    data_filename = "PreCo_1.0/dev.jsonl"
    mention_method = "extract"


    data = load(data_filename, mention_method=mention_method)

    result = model_baseline(data)
    display_coreferences(data[0], result[0])
    print(evaluate_b3(data[0], result[0]))
    print(evaluate_b3(data[0], data[0].real_coreferences))


if __name__ == "__main__":
    main()