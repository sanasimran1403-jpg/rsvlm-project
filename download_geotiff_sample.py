from datasets import load_dataset

ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)
sample = next(iter(ds))
sample["img"].save("data/sample_bigearthnet.tif")
print("saved GeoTIFF sample")
print("labels:", sample["labels"])
