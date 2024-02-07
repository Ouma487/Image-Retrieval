# Article Identification And Image Retrieval

This project aims to identify the reference of an article from a photo of it using the visual characteristics of the objects. It relies on the Dior dataset, composed of two directories containing reference images for each article and test images.

## Data

### data/DAM/
This directory contains JPEG images that serve as reference images for each article in the dataset. There are 2,766 articles in total, and each JPEG file is named based on its MMC (Manufacturer's Material Code) referenced in the accompanying CSV file. The size of each image is standardized to 256x256 pixels.

### test_image_headmind/
This directory contains test images used to evaluate the model's performance. There are 80 test images in total. All articles included in these images are also referenced in the DAM directory and the accompanying CSV file. The size of these images varies, and they are not annotated. The filenames correspond to the nomenclature given by the camera.

### product_list.csv
This CSV file contains two main columns:

- MMC (Manufacturer's Material Code): A unique code assigned to each article in the dataset.
- Product_BusinessUnitDesc: This column specifies the class of the article, such as Bags, Shoes, etc.

## Objective

The main objective of this project is to develop a model capable of identifying the reference of an article from a photo of it. To achieve this, we will implement a strategy based on article classification and reference attribution.

## Strategy

The strategy adopted to achieve the project's objective is based on two main axes:

1. **Classification by Model Type**: Utilization of different CNN architectures to classify articles.
2. **Attribution of the Closest Model**: Determination of the closest model to identify the article's reference.


## Files

### utils 

The `utils` directory contains utility scripts & modules used across other scripts and notebooks in this repository.

### Data_exploration.ipynb

`Data_exploration.ipynb` is a Jupyter notebook dedicated to initial data analysis and visualization. It covers steps such as loading data, checking for missing values, visualizing data distributions, and conducting preliminary statistical analysis. This notebook provides insights into the dataset before proceeding with further analysis or modeling tasks.

### Segmentation.ipynb

`Segmentation.ipynb` is a Jupyter notebook focused on image segmentation tasks. It contains code and explanations for segmenting images into different regions.

### complete_pipeline_1.ipynb

`complete_pipeline_1.ipynb`  contains a complete pipeline for classification and the search for the closest article models. This notebook includes model training, evaluation, and possibly integrating the results with some application, such as fetching the closest article related to the classification result. 

## Contribution

Contributions to this project are welcome. If you would like to contribute, please submit a pull request with a clear description of the changes made.

## License

This project is licensed under the [MIT License](LICENSE).

---
