import argparse

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSpeech model on TPU using librispeech dataset"
    )
    # Loader args
    parser.add_argument(
        "--input-file", default="input.txt", help="Specify the full path to txt file")
    parser.add_argument(
        "--print-length", type=int, default=100, help="Length of a file for preprinting")
    parser.add_argument(
        "--output-file", default="output.txt", help="Length a name for an output file")
    args = parser.parse_args()
    return args


def analyze_syntax(text):
    doc = nlp(text)

    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    df = pd.DataFrame([{'text': e.text, 'label': e.label_} for e in doc.ents])
    return df


def main():
    args = parse_args()
    print(args)
    print('\n')
    with open(args.input_file) as f:
        text = ' '.join([line.strip() for line in f.readlines()])
        print(text[:args.print_length], '...')
        print('\n')

    df = analyze_syntax(text)
    print(df)
    df.to_csv(args.output_file, index=False)
    print(f"Saved {args.output_file}")


if __name__ == '__main__':
    main()
