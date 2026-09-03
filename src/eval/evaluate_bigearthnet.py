"""
Phase 2: Evaluation script for LoRA fine-tuned Qwen2-VL on BigEarthNet VQA.
Computes exact-match rate and average label overlap against ground truth.
Run this on Colab (GPU required) after loading the fine-tuned model.
"""
import json


def run_inference(image_path, question, model, processor, process_vision_info):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_space=True
    )
    return output_text[0]


def evaluate(val_json_path, model, processor, process_vision_info, n_eval=20):
    val_data = json.load(open(val_json_path))

    exact_matches = 0
    label_overlap_scores = []
    results = []

    for i in range(min(n_eval, len(val_data))):
        sample = val_data[i]
        pred = run_inference(sample["image"], sample["question"], model, processor, process_vision_info)

        true_labels = set(sample["raw_labels"])
        matched = set([l for l in true_labels if l.lower() in pred.lower()])
        overlap = len(matched) / len(true_labels) if true_labels else 0
        label_overlap_scores.append(overlap)

        if overlap == 1.0:
            exact_matches += 1

        results.append({
            "index": i,
            "true_labels": sample["raw_labels"],
            "prediction": pred,
            "overlap": overlap,
        })
        print(f"[{i}] true={sample['raw_labels']} | overlap={overlap:.2f}")

    exact_match_rate = exact_matches / n_eval
    avg_overlap = sum(label_overlap_scores) / len(label_overlap_scores)

    print(f"\nExact match rate: {exact_matches}/{n_eval} = {exact_match_rate:.2%}")
    print(f"Avg label overlap: {avg_overlap:.2%}")

    return {
        "exact_match_rate": exact_match_rate,
        "avg_label_overlap": avg_overlap,
        "results": results,
    }
