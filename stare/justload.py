from wildlife_datasets.datasets import MacaqueFaces
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
import torch
import torchvision.models as models
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata pointing to transformed images
metadata_transformed = MacaqueFaces('data/MacaqueFacesTransformed')

# Define transform if necessary
transform = None  # Or only include necessary transforms as shown earlier

# Define dataset split
my_split = 170
dataset_database = WildlifeDataset(
    metadata_transformed.df.iloc[my_split:, :],
    metadata_transformed.root,
    transform=transform
)
dataset_query = WildlifeDataset(
    metadata_transformed.df.iloc[:my_split, :],
    metadata_transformed.root,
    transform=transform
)

# Load your custom model
model_architecture = models.resnet50(num_classes=0)  # Adjust as per your model
extractor = DeepFeatures(model_architecture)
extractor.model.load_state_dict(torch.load('pytorch_model.bin', map_location=device))
extractor.model.to(device)

# Feature extraction
query = extractor(dataset_query)
database = extractor(dataset_database)

# Similarity computation
similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)

# Classification and accuracy
classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
predictions = classifier(similarity['cosine'])
accuracy = np.mean(dataset_query.labels_string == predictions)
print(f"Accuracy: {accuracy}")
