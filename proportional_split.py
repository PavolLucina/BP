import numpy as np

# Group by lynx ID and split proportionally
def proportional_split(metadata_df, query_ratio=0.2, seed=42):
    query_indices = []
    database_indices = []
    grouped = metadata_df.groupby("identity")  # Assuming "ID" is the lynx identifier

    for _, group in grouped:
        num_images = len(group)
        num_query = max(1, int(np.ceil(num_images * query_ratio)))  # Ensure at least one image per lynx
        shuffled_indices = group.sample(frac=1, random_state=seed).index  # Shuffle images per lynx

        query_indices.extend(shuffled_indices[:num_query])  # Take first N for query
        database_indices.extend(shuffled_indices[num_query:])  # Remaining go to database

    return metadata_df.loc[database_indices], metadata_df.loc[query_indices]


