# Dataset cache

Runtime downloads are cached under `data/cache/`. The cache directory is excluded from version control because the datasets are maintained by their original sources and can be large.

The loaders fetch SECOM through UCI, C-MAPSS through the configured public mirror when needed, Tennessee Eastman reference files directly from its repository, and FactoryNet through Hugging Face streaming. Missing sources raise an explicit error; synthetic replacements are not generated.
