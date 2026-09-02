from datasets import load_dataset

ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)
sample = next(iter(ds))
print(sample.keys())
print({k: (v if k != "img" else "IMAGE_DATA") for k, v in sample.items()})
