from datasets import load_dataset

ds = load_dataset("xiang709/VRSBench", streaming=True)
print(ds)
