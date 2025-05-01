#!/usr/bin/env python3

import json
import argparse
from itertools import product

def extract_text(text, span):
    return text[span["start"]:span["end"]]

def create_input_with_markers(text, head_span, tail_span):
    # Add entity markers to the text
    spans = sorted([head_span, tail_span], key=lambda x: x["start"])
    marked = (
        text[:spans[0]["start"]] +
        ("<e1>" if spans[0] == head_span else "<e2>") +
        text[spans[0]["start"]:spans[0]["end"]] +
        ("</e1>" if spans[0] == head_span else "</e2>") +
        text[spans[0]["end"]:spans[1]["start"]] +
        ("<e2>" if spans[1] == tail_span else "<e1>") +
        text[spans[1]["start"]:spans[1]["end"]] +
        ("</e2>" if spans[1] == tail_span else "</e1>") +
        text[spans[1]["end"]:]
    )
    return marked

def generate_re_examples(data):
    examples = []
    for sample in data:
        text = sample["text"]
        entities = sample["entities"]
        relations = sample.get("relations", [])

        rel_dict = {(r["source"], r["target"]): r["type"] for r in relations}

        for i, j in product(range(len(entities)), repeat=2):
            if i == j: continue

            head = entities[i]
            tail = entities[j]
            relation = rel_dict.get((i, j), "no_relation")

            input_text = create_input_with_markers(text, head, tail)

            example = {
                "sentence": text,
                "head": extract_text(text, head),
                "tail": extract_text(text, tail),
                "head_type": head["label"],
                "tail_type": tail["label"],
                "relation": relation,
                "input": f"[CLS] {input_text} [SEP]"
            }
            examples.append(example)

    return examples

def main(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)

    examples = generate_re_examples(data)

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"[✓] Generated {len(examples)} relation examples → {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BERT-compatible RE dataset.")
    parser.add_argument("--input", required=True, help="Path to annotated JSON")
    parser.add_argument("--output", required=True, help="Path to save RE dataset")

    args = parser.parse_args()
    main(args.input, args.output)
