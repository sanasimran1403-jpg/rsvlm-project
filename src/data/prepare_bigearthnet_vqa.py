"""
Phase 2: Converts BigEarthNet multi-label data into VQA-style
instruction-tuning format for LoRA fine-tuning of Qwen2-VL.
"""
from datasets import load_dataset
import json
import os

QUESTION_TEMPLATES = [
    "What land cover types are present in this image?",
    "Describe the land use categories visible in this satellite image.",
    "What classes of land cover can be identified here?",
]


def format_answer(labels):
    """Turn label list into a natural sentence."""
    if len(labels) == 1:
        return f"This image shows {labels[0]}."
    return "This image shows " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def build_dataset(split="train", num_samples=2000, output_path=None):
    ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split=split, streaming=True)

    records = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break

        img_path = f"/content/rsvlm-project/data/bigearthnet_images/{split}_{i}.jpg"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        sample["img"].save(img_path)

        question = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        answer = format_answer(sample["labels"])

        records.append({
            "image": img_path,
            "question": question,
            "answer": answer,
            "raw_labels": sample["labels"],
        })

        if (i + 1) % 500 == 0:
            print(f"processed {i + 1} samples")

    if output_path is None:
        output_path = f"data/bigearthnet_vqa_{split}.json"

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    build_dataset(split="train", num_samples=n)
