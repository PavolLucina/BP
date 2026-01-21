from wildlife_datasets.datasets import MacaqueFaces
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
import torch
import timm
import torchvision.models as models
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import numpy as np
from sklearn.model_selection import train_test_split
# from test import dataset

# Device configuration
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # exit()

    # Load metadata pointing to transformed images
    metadata_transformed = MacaqueFaces('data_makaky/MacaqueFacesTransformed')

    # Define transform with ToTensor and Normalize only
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Define dataset split
    my_split = 170
    # test, train =
    print(metadata_transformed.df[["identity", "path"]])
    print(list(metadata_transformed.df))

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

    # print(dataset_database.metadata)
    # print(len(dataset_database.labels_string))
    # print(dataset_query.metadata)

    # Load your custom model
    # model_architecture = models.resnet50(num_classes=0)  # Adjust as per your model
    # extractor = DeepFeatures(model_architecture)
    # extractor.model.load_state_dict(torch.load('pytorch_model.bin', map_location=device))
    # extractor.model.to(device)

    name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True), device=str(device))

    # Feature extraction
    query = extractor(dataset_query)
    database = extractor(dataset_database)
    print(database)
    print(database.data)
    # Similarity computation
    # prerobit na najpodobnejsi obrazok
    # vyskusat na ciernobielych obrazkoch
    similarity_function = CosineSimilarity()
    similarity = similarity_function(query, database)
    print(similarity)

    # Classification and accuracy
    classifier = KnnClassifier(k=5, database_labels=dataset_database.labels_string)
    predictions = classifier(similarity['cosine'])
    print(predictions)

    accuracy = np.mean(dataset_query.labels_string == predictions)
    print(f"Accuracy: {accuracy}")
