# https://github.com/WildlifeDatasets/wildlife-datasets

from wildlife_datasets.datasets import MacaqueFaces
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
import os

metadata = MacaqueFaces('data/MacaqueFaces')
transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset = WildlifeDataset(metadata.df, metadata.root, transform=transform)
unique_categories = metadata.df['category'].unique()
unique_identities = [
    'Dan', 'Judd', 'Lala', 'Leah', 'Libby', 'Linz', 'Love', 'Lydia', 'Maj', 'Meesha', 'Meg',
    'Melody', 'Mindy', 'Ocelot', 'Rupee', 'Saphy', 'Serena', 'Shirley', 'Sizzle', 'Sol', 'Sonja',
    'Spice', 'Star', 'Sugar', 'Tamara', 'Tass', 'Tea', 'Teal', 'Tes', 'Thyme', 'Umbrella',
    'Ursula', 'Venus', 'Verity'
]

base_save_dir = 'data/MacaqueFacesTransformed'
os.makedirs(base_save_dir, exist_ok=True)
for category in unique_categories:
    for identity in unique_identities:
        identity_dir = os.path.join(base_save_dir, category, identity)
        os.makedirs(identity_dir, exist_ok=True)

print("All subdirectories created.")

# Iterate over the dataset and save the images in the pre-created directories
unnormalize = T.Compose([
        T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.ToPILImage()
    ])
for idx, (image, _) in enumerate(dataset):
    category = metadata.df.iloc[idx]['category']
    identity = metadata.df.iloc[idx]['identity']

    # Save the transformed image in the corresponding pre-created directory
    identity_dir = os.path.join(base_save_dir, category, identity)
    img_pil = unnormalize(image)
    img_pil.save(os.path.join(identity_dir, f"image_{idx}.png"))

print("Images saved into pre-created directories.")
