from wildlife_datasets.datasets import Lynx
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
import torch
import timm
import torchvision.models as models
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import numpy as np
from proportional_split import proportional_split

# Device configuration
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # exit()

    # Load metadata pointing to transformed images
    metadata_transformed = Lynx('data_rysy/rys_trening_data_Beno')

    # Define transform with ToTensor and Normalize only
    transform = T.Compose([
        T.Resize([224,224]),
        # transform to grayscale here
        # T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Define dataset split
    dataset_database_df, dataset_query_df = proportional_split(metadata_transformed.df, query_ratio=0.2)
    print(metadata_transformed.df.values)
    dataset_database = WildlifeDataset(
        dataset_database_df,
        metadata_transformed.root,
        transform=transform
    )
    dataset_query = WildlifeDataset(
        dataset_query_df,
        metadata_transformed.root,
        transform=transform
    )
    print(dataset_database)
    print(metadata_transformed.df)
    print(dataset_query.labels_string)

    # Load your custom model
    # model_architecture = models.resnet50(num_classes=0)  # Adjust as per your modela
    # extractor = DeepFeatures(model_architecture)
    # extractor.model.load_state_dict(torch.load('pytorch_model.bin', map_location=device))
    # extractor.model.to(device)

    name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True), batch_size=1, device=str(device))

    # Feature extraction
    query = extractor(dataset_query)
    database = extractor(dataset_database)

    # Similarity computation
    # prerobit na najpodobnejsi obrazok
    # pridavat po jednom z kazdeho do trenovacej mnoziny
    similarity_function = CosineSimilarity()
    similarity = similarity_function(query, database)
    print(similarity)

    # Classification and accuracy
    classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
    predictions = classifier(similarity['cosine'])
    print(predictions)
    accuracy = np.mean(dataset_query.labels_string == predictions)
    print(f"Accuracy: {accuracy}")

    unique_classes = np.array(['Adam', 'Albin', 'Benadik', 'Brano', 'Dio', 'Edo', 'Eliska', 'Izidor', 'Kiara',
                             'Lubos', 'Milos', 'Roman', 'Silvester', 'Zora'])

    image_classes = np.array(metadata_transformed.df.values[:, 1])  # Shape: (321, 4)

    # Get cosine similarity matrix
    cosine_sim = similarity["cosine"]  # Shape: (n_predictions, 321)

    # Get top 5 highest values for each row
    top5_indices = np.argsort(-cosine_sim, axis=1)[:, :5]  # Sort in descending order, take top 5 indices

    # Retrieve similarity scores for those indices
    top5_scores = np.take_along_axis(cosine_sim, top5_indices, axis=1)
    top5_classes = image_classes[top5_indices]
    for i, (classes, scores) in enumerate(zip(top5_classes, top5_scores)):
        print(f"Image {dataset_query_df[i]}:")
        for cls, score in zip(classes, scores):
            print(f"  {cls}: {score:.4f}")
        print()

    # Map indices to actual class names using image_classes
    top5_classes = image_classes[top5_indices]


