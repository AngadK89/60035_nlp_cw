# 60035: Natural Language Processing Coursework

This repository contains the implementation for **Task 4: Patronising and Condescending Language Detection** from **SemEval 2022 Task 4 (Part 1)**.  

Task description and shared task details can be found here:  
https://sites.google.com/view/pcl-detection-semeval2022/


## Project Structure

This repository is organised as follows:


### Exploratory Data Analysis (EDA)
- All exploratory data analysis is contained in the [`eda/`](./eda) folder:
  - [`eda/eda.ipynb`](./eda/eda.ipynb)
  - Supporting resources (e.g., stopword lists)

### Final Model
- The final model implementation and training pipeline are provided in [`stage4.ipynb`](./stage4.ipynb)

### Inference Outputs
- Predictions for the **dev** and **test** sets are located under the [`inference/`](./inference) folder:
  - [`inference/dev.txt`](./inference/dev.txt)
  - [`inference/test.txt`](./inference/test.txt)
- The inference pipeline notebook is available at [`inference/inference.ipynb`](./inference/inference.ipynb)

### Error Analysis
- All error analysis is conducted in the [`error_analysis/`](./error_analysis) folder  
  *(to be uploaded)*
