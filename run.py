import argparse

from src import evaluate, predict, train, summary


def main():
    parser = argparse.ArgumentParser(
        description="Scientific Paper Categorizer - Pipeline Runner"
    )
    parser.add_argument(
        "--task", choices=["train", "evaluate", "predict", "summary"], required=True
    )
    parser.add_argument(
        "--method",
        choices=["tfidf", "sbert"],
        default="tfidf",
        help="Feature method for train/predict",
    )
    parser.add_argument("--abstract", type=str, help="Abstract string for prediction")

    args = parser.parse_args()

    if args.task == "train":
        print("🚀 Training model...")
        train.main(method=args.method)

    elif args.task == "evaluate":
        print("📊 Evaluating model...")
        evaluate.main()

    elif args.task == "predict":
        model_map = {
            "tfidf": "tfidf_logistic_regression.pkl",
            "sbert": "sbert_logistic_regression.pkl",
        }
        model_name = model_map.get(args.method, "tfidf_logistic_regression.pkl")
        result = predict.predict_abstract_labels(
            abstract=args.abstract, model_name=model_name, method=args.method
        )
        print(f"📄 Predicted Categories: {', '.join(result)}")
        
    elif args.task == "summary":
        print("📊 Generating performance summary...")
        summary.main()


if __name__ == "__main__":
    main()
