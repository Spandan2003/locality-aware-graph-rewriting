from data_loader import load_lrgb_dataset

# Try loading peptides-functional
loader = load_lrgb_dataset('peptides-functional', split='train', batch_size=4)

for batch in loader:
    print(batch)
    break