"""
Phase 2: LoRA fine-tuning of Qwen2-VL on BigEarthNet VQA data.
Run this on Colab (GPU required).
"""
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model


class BigEarthNetVQADataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path) as f:
            self.records = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rec["image"]},
                    {"type": "text", "text": rec["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec["answer"]}],
            },
        ]
        return messages


def collate_fn(batch, processor):
    texts = []
    image_inputs_all = []
    for messages in batch:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        image_inputs, _ = process_vision_info(messages)
        image_inputs_all.extend(image_inputs)

    inputs = processor(
        text=texts, images=image_inputs_all, padding=True, return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def main():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    min_pixels = 256 * 28 * 28
    max_pixels = 640 * 28 * 28

    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = BigEarthNetVQADataset("/content/rsvlm-project/data/bigearthnet_vqa_train.json", processor)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    num_epochs = 1
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if step % 500 == 0 and step > 0:
                model.save_pretrained(f"/content/rsvlm-project/checkpoints/lora_epoch{epoch}_step{step}")
                print(f"checkpoint saved at step {step}")

    model.save_pretrained("/content/rsvlm-project/checkpoints/lora_final")
    print("training complete, final adapter saved")


if __name__ == "__main__":
    main()
