from datasets import load_dataset

ds = load_dataset("xiang709/VRSBench", streaming=True)
sample = next(iter(ds["train"]))
print({k: (v if k != "image" else "IMAGE_DATA") for k, v in sample.items()})
