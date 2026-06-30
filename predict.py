import pickle
import sys

from features import extract_features


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)

    scaler = bundle["scaler"]
    clf = bundle["clf"]

    feats = extract_features(image_path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    prob_screen = clf.predict_proba(feats_scaled)[0, 1]

    print(round(float(prob_screen), 4))


if __name__ == "__main__":
    main()
