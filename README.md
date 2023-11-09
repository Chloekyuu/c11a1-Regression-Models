# CSCC11 A1 - Regression Models
> [Part 1: Exploring a Combination of Clustering and Linear Regression](https://www.notion.so/A1-Regression-Models-6ccf5654addc4ceaa9309b18a712dd44?pvs=21)
> 
> 
> [Part 2: Image Inpainting with RBF Regression](https://www.notion.so/A1-Regression-Models-6ccf5654addc4ceaa9309b18a712dd44?pvs=21)
> 

## Part 1: Exploring a Combination of Clustering and Linear Regression

### Introduction

Predict the chances of students' admission to a university of rank # based on their academic backgrounds, including GRE scores, TOEFL scores, CGPA, and research experience. Two approaches are explored: a single linear regression model and clustering-based linear regression.

The graduate admissions dataset is used for this experiment.

### 1.1 Applying Single Linear Regression Model

1. **Read** the Admission_Predict.csv file into a data frame.
2. **Split** the data into training (70%) and testing (30%) subsets.
3. Compute the closed form solution for **linear regression**.
4. Provide **predictions** on the test data.
5. Define **Mean Absolute Error (MAE)** & **Mean Squared Error (MSE)** and explain their relevance.
6. Find optimal parameters using training data and estimate **Chance of Admit** values for testing data. Report MAE and MSE.
    
    > Mean squared error is more preferred because it is more sensitive to the outliners. It can better detect outliners. We squared the error, which means if a data point has larger error (i.e. is an outliner), it will have more weight on the final result (i.e. the mean error) for the whole data set.
    > 

### 2.2 Clustering-Based Linear Regression

1. Calculate Silhouette Coefficient for K (# of clusters) ranging from 2 to 10. Determine the most appropriate value.
    
    > 2 will be the most appropriate.
    > 
2. Apply **K-Means clustering** with the determined number of clusters on the training dataset.
3. Build **separate** linear regression models for each cluster.
    
    > Different input and arguments for K-Means will result in a different error outcome. For example, if there are more outliners in our input data, the error will get larger.
    > 

### Files and Requirements

- Find the starter Code for Part 1 in `StarterCode/Part1`
    - `K_means_LP.ipynb`: Contains starter code for Part 1 of the assignment.
    - `Admission_Predict.csv`: The dataset used for the project.
- Install the required libraries: `pip install pandas matplotlib numpy scikit-learn scipy`

## Part 2: Image Inpainting with RBF Regression

### Introduction

Inpainting is the process of predicting the corrupted pixels based on the surrounding information in the image. This project is aim to predict the missing or corrupted pixels by finding a smooth Radial Basis Function (RBF) regression model that accurately represents brightness as a function of position in the image.

### Optimized Approach

1. **2D RBF Regression:** Implemented the `RBFRegression` class with methods for computing RBF outputs, predicting values, and finding regularized least squares solutions.
2. **Inpainting Process:** Utilized the RBF regression model to inpaint images, effectively removing corrupted text and restoring visual clarity.
3. **Hyperparameter Exploration:** Conducted extensive experimentation with hyperparameters, including RBF spacing, width, and regularization constant, to optimize inpainting results.

### Additional Notes

- The starter code provides basic checks for correctness, but thorough testing and parameter tuning are essential for optimal results.
    
    > The image is not fixed well when we set `spacing=1` , but other values from 2 to 9 seems not much difference. The change of width will change the output image. Therefore, we need to test to find out the optimal value of RBF width.
    > 
- Experiment with hyperparameters to understand their impact on the inpainting process.
    
    > By increasing the amount of hyper-parameters, it is possible to build a better model. For example add a smooth term *λ*, we can avoid overfitting. However, we need to set the appropriate value of our hyper-parameters.
    > 

### Files and Requirements

- Find the starter code for Part 2 in `StarterCode/Part2`:
    - **`rbf_regression.py`**: contains the class `RBFRegression` with methods for computing RBF outputs and performing regression.
        - `__init__(self, centers, widths)`: Constructor for initializing RBF centers and widths.
        - `_rbf_2d(self, X, rbf_i)`: Computes the output of the ith RBF given a set of 2D inputs.
        - `predict(self, X)`: Predicts the output for the given inputs using the model parameters.
        - `fit_with_l2_regularization(self, train_X, train_Y, l2_coef)`: Finds the regularized LS solution.
    - **`rbf_image_inpainting.py`**: Demonstrates the application of RBF regression for image inpainting. Allows customization of various hyperparameters.
- Install the required libraries: `pip install matplotlib numpy turtle`
