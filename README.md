# Activity Recognition and Rep Analysis ML Pipeline

## 📖 In-Depth Project Overview
This repository documents an end-to-end machine learning pipeline designed to create a sophisticated fitness tracking system. The primary goal is to leverage raw sensor data from a wrist-worn wearable device (containing an accelerometer and a gyroscope) to build a system capable of two core functions:

1. **Automatic Exercise Classification**: 
   Accurately identifying which of five fundamental barbell exercises a user is performing:  
   - Bench Press  
   - Squat  
   - Deadlift  
   - Overhead Press (OHP)  
   - Row
  
2. **Automated Repetition Counting**: Precisely counting the number of repetitions completed within each exercise set.

The project meticulously follows a structured data science lifecycle, beginning with raw data ingestion and culminating in the deployment and evaluation of highly accurate machine learning models. It serves as a comprehensive case study in time-series analysis, signal processing, feature engineering, and predictive modeling. The final classification model achieves an impressive 99%+ accuracy, and the repetition counting algorithm demonstrates high precision, making this a robust foundation for a real-world, context-aware fitness application.

## 🛠️ Technology Stack & Environment Setup
This project is built entirely in Python and relies on a suite of powerful libraries for data science and machine learning. A Conda environment is used to ensure reproducibility.

1. **Programming Language**: Python 3.8+
  
2. **Core Libraries**:
    - Pandas: For all data manipulation, cleaning, and structuring.
    - NumPy: For high-performance numerical computations.
    - Scikit-learn: The cornerstone for machine learning, used for model training, grid search, evaluation metrics, and preprocessing.
    - SciPy: Leveraged for advanced signal processing tasks, including the Butterworth filter and peak detection (argrelextrema).
    - Matplotlib & Seaborn: For creating all static visualizations, from initial exploratory plots to final results dashboards.
    - Environment Management: Conda is used to manage packages and create an isolated, reproducible environment.

## 🚀 The Data Science Pipeline: A Step-by-Step Breakdown
The project is organized into a modular pipeline, with each script performing a distinct, critical task. The scripts are designed to be run sequentially, with the output of one step serving as the input for the next.

### Phase 1: Data Ingestion and Processing (src/data/make_dataset.py)
**Goal**: To take the numerous raw, disparate CSV files from the sensor and consolidate them into a single, clean, and uniformly-timed DataFrame.

**Detailed Process**:

1. **File Discovery**: The script begins by using Python's glob library to find all .csv files within the data/raw/MetaMotion/ directory.

2. **Metadata Extraction from Filenames**: Each filename contains crucial metadata (participant ID, exercise label, and weight category). The script programmatically parses each filename to extract these labels. For example, a file named A-bench-heavy_...csv is broken down to identify participant A, exercise bench, and category heavy.

3. **Iterative Reading and Merging**: The script iterates through every discovered file.
    - It reads the CSV into a temporary Pandas DataFrame.
    - The extracted metadata (participant, label, category) is added as new columns to this DataFrame.
    - A unique set ID is assigned to each exercise performance.
    - Based on whether the filename indicates "Accelerometer" or "Gyroscope", the data is appended to one of two master DataFrames: acc_df or gyr_df.

4. **Time-based Indexing**: The raw data includes an epoch timestamp (in milliseconds). This column is converted to a proper datetime object using pd.to_datetime and set as the index for both the accelerometer and gyroscope DataFrames. This transforms the data into a time-series format, which is essential for the next steps. Redundant time-related columns are dropped.

5. **Final Merging**: The processed accelerometer and gyroscope DataFrames are concatenated column-wise (axis=1). Since their datetime indices now align, this merge creates a single row for each timestamp containing all six sensor readings (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z).

6. **Resampling for Uniform Frequency**: The raw accelerometer and gyroscope data were recorded at different frequencies (12.5Hz and 25Hz, respectively). To create a uniform time series, the merged DataFrame is resampled to a fixed frequency of 200ms (5Hz) using df.resample('200ms').apply(sampling). During resampling, numerical sensor values are aggregated using their mean, while categorical labels are propagated using last. This step is crucial for preventing data leakage and ensuring consistency.

7. **Exporting Processed Data**: The final, cleaned DataFrame is saved as a pickle file (01_data_processed.pkl) in the data/interim/ directory, ready for the next phase.

### Phase 2: Exploratory Data Analysis & Visualization (src/visualization/visualize.py)
**Goal**: To visually inspect the processed data to understand its structure, identify patterns, and uncover relationships between different variables.

**Detailed Process**:

1. **Loading Processed Data**: The script loads the 01_data_processed.pkl file.

2. **Plotting Individual Signals**: Initial plots are generated for single sensor axes (e.g., acc_y) for a specific set. This helps to get a first impression of the signal's nature.

3. **Comparing Exercises**: The script loops through each unique exercise label and plots the sensor data. This reveals the distinct "fingerprints" of each movement. For instance, the cyclical pattern of a bench press looks vastly different from the sharp, singular motion of a deadlift.

4. **Comparing Participants and Categories**:
    - Visualizations are created to compare the same exercise (bench) across all participants. This helps assess inter-subject variability.
    - Plots are also generated to compare medium vs. heavy sets for the same exercise, revealing differences in execution speed and force.

5. **Multi-Axis Visualization**: To get a holistic view, the script plots the X, Y, and Z axes for a single sensor (e.g., accelerometer) on the same graph. This shows how the different directional forces interact during an exercise.

6. **Combined Sensor Plots**: The final and most informative visualization combines the accelerometer and gyroscope plots into a single figure with two subplots. This allows for a direct comparison of linear acceleration and rotational velocity for any given exercise and participant.

7. **Exporting Figures**: All generated plots are systematically saved as PNG files to the reports/figures/ directory for use in documentation and analysis.

### Phase 3: Outlier Detection and Removal (src/features/remove_outliers.py)
**Goal**: To identify and handle anomalous data points (outliers) that could negatively impact model performance.

**Detailed Process**:

1. **Method Comparison**: Three different outlier detection techniques were implemented and visually compared:
    - Interquartile Range (IQR): A statistical method that defines outliers as points falling outside 1.5 times the IQR below the first quartile or above the third quartile.
    - Local Outlier Factor (LOF): A distance-based method that identifies outliers based on the local density of data points.
    - Chauvenet's Criterion: A probabilistic method that rejects data points if their probability of occurrence is very low, assuming a normal distribution.

2. **Visual Validation**: For each method, a custom function plot_binary_outliers was used to visualize the identified outliers in red against the non-outlier data points in blue. This allowed for a qualitative assessment of each method's effectiveness.

![image](https://github.com/user-attachments/assets/4bc16071-fdce-4ab4-8469-1cc0a616ec4d)
![image](https://github.com/user-attachments/assets/abe84baa-acd4-402d-807a-b24f5ac67c86)

3. **Method Selection**: After visual inspection, Chauvenet's Criterion was chosen as the most suitable method. It provided a good balance, effectively identifying extreme, isolated spikes without being overly aggressive and flagging parts of legitimate exercise movements, which the IQR method tended to do.

4. **Grouped Outlier Removal**: A crucial decision was made to apply the outlier detection not on the entire dataset at once, but grouped by exercise label. This prevents data from one type of exercise (e.g., a high-impact squat) from being incorrectly flagged as an outlier when compared to a low-impact exercise (e.g., rest).

5. **Handling Outliers**: Instead of dropping the entire row, the identified outlier values were replaced with NaN (Not a Number). This preserves the time-series structure and allows the missing values to be handled more intelligently in the next phase (imputation).

6. **Exporting Cleaned Data**: The DataFrame with outliers replaced by NaN is saved as a new pickle file: 02_outliers_removed_chauvenet.pkl.

### Phase 4: Advanced Feature Engineering (src/features/build_features.py)
**Goal**: To transform the cleaned, time-series data into a rich feature set that captures complex patterns, thereby enhancing the predictive power of the machine learning models.

**Detailed Process**:

1. **Missing Value Imputation**: The NaN values created during outlier removal are filled using linear interpolation (df.interpolate()). This is an ideal method for time-series data as it estimates a missing value based on the values immediately before and after it, preserving the signal's trend.

2. **Signal Smoothing (Butterworth Low-Pass Filter)**: A low-pass filter is applied to each of the six sensor channels. This removes high-frequency jitter and noise, resulting in a smoother signal that emphasizes the core movement pattern of the exercise. The cutoff frequency was carefully selected to filter noise without distorting the underlying repetition pattern.

3. **Dimensionality Reduction (Principal Component Analysis - PCA)**:
    - PCA is applied to the six sensor columns to create new, composite features (pca_1, pca_2, pca_3).
    - The "elbow method" was used to determine that three principal components were optimal, capturing most of the variance in the original six features without adding redundant information. PCA helps in creating powerful, uncorrelated features.

4. **Creating Magnitude Features (Sum of Squares)**:
    - New features, acc_r and gyr_r, are created by calculating the vector magnitude (Euclidean norm) of the accelerometer and gyroscope readings, respectively:  √(x² + y² + z²).
    - These magnitude features are orientation-invariant, meaning they capture the overall intensity of movement regardless of the sensor's specific orientation on the wrist.

5. **Temporal Abstraction (Rolling Window Features)**:
    - For each sensor feature (including the new magnitude features), rolling window statistics are calculated.
    - Specifically, the mean and standard deviation over a 5-sample window (1 second) are computed. This creates features like acc_y_temp_mean_ws_5.
    - These features provide the model with context about the signal's recent trend and variability.

6. **Frequency Domain Features (Fourier Transform)**:
    - A Fast Fourier Transform (FFT) is applied to the data in windows to decompose the time-series signal into its constituent frequencies.
    - This is the most powerful feature engineering step, generating a large set of features for each sensor axis, including the dominant frequency, weighted average frequency, and spectral entropy. These features are exceptionally good at capturing the unique rhythmic patterns of different exercises.

7. **Dealing with Overlapping Windows**: The rolling window and frequency analysis steps introduce high correlation between consecutive rows. To mitigate this and reduce the risk of overfitting, every other row of the dataset is dropped, effectively reducing the window overlap to 50%.

8. **Clustering for Pattern Discovery (K-Means)**:
    - The K-Means clustering algorithm is applied to the accelerometer columns (acc_x, acc_y, acc_z) to group data points into clusters based on their similarity.
    - The "elbow method" was used again to determine the optimal number of clusters, which was found to be five.
    - A new cluster feature is added to the DataFrame. This unsupervised learning step helps the model by providing it with a high-level feature that represents distinct types of motion patterns (e.g., one cluster might represent the "bottom" of a squat, while another represents the "top").

9. **Exporting Feature-Rich Data**: The final DataFrame, now containing 119 powerful features, is saved as 03_data_features.pkl.

### Phase 5: Predictive Modeling & Evaluation (src/models/train_model.py)
**Goal**: To train, compare, and fine-tune various classification models to find the best performer for the exercise recognition task.

**Detailed Process**:

1. **Data Splitting (Stratified)**: The feature-rich dataset is split into a training set (75%) and a testing set (25%) using train_test_split. The stratify=y parameter is used to ensure that the distribution of exercise labels is the same in both the training and testing sets, which is crucial for a balanced evaluation.

2. **Feature Set Creation**: Four distinct feature sets are created to systematically evaluate the impact of the engineered features. A fifth set is created using the best features from a forward selection process.

3. **Forward Feature Selection**: A simple Decision Tree is used to iteratively select the top 10 most predictive features from the entire feature set. This demonstrated that frequency-domain features were by far the most impactful.

4. **Grid Search for Hyperparameter Tuning**: A comprehensive grid search (GridSearchCV) is performed for five different classification algorithms:

![image](https://github.com/user-attachments/assets/0f77deff-47d8-4606-bc56-243262f26c71)
![image](https://github.com/user-attachments/assets/22b20cdd-c0b4-4f24-a1bd-caaa6b365c6d)

    - Random Forest
    - Feedforward Neural Network
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Naive Bayes
    
    The grid search systematically tests various combinations of hyperparameters for each model, using 5-fold cross-validation to find the optimal settings that prevent overfitting.

5. **Model Comparison**: The performance (accuracy) of each of the five models is evaluated across the different feature sets. The results are visualized in a bar plot, clearly showing that the Random Forest model consistently performed the best, especially with the full feature set (Feature Set 4).

6. **Final Model Evaluation**:
    - The best model (Random Forest with all features) is re-trained.
    - A confusion matrix is generated to provide a detailed breakdown of its performance. This reveals which exercises, if any, the model tends to confuse (e.g., a few instances of 'row' were misclassified as 'deadlift').

7. **Generalization Test (Participant-based Split)**: To perform the ultimate test of the model's ability to generalize, a new train-test split is created. The data from one participant (A) is entirely held out as the test set, while the model is trained on the data from all other participants. The Random Forest model maintained its ~99.4% accuracy, proving that it can accurately classify exercises for new users not seen during training.

### Phase 6: Repetition Counting (src/features/count_repetitions.py)
**Goal**: To develop and benchmark a custom algorithm for counting repetitions within an exercise set.

**Detailed Process**:

1. **Data Preparation**: The script loads the processed data (before heavy feature engineering) and calculates the sum-of-squares magnitude feature (acc_r), which provides a clean, orientation-invariant signal for counting.

2. **Signal Filtering**: A Butterworth low-pass filter is once again applied to the chosen signal (acc_r or gyr_x depending on the exercise). The filter's cutoff frequency is carefully tuned for each specific exercise to maximize the prominence of the peaks corresponding to repetitions.

3. **Peak Detection**: The core of the algorithm uses the argrelextrema function from scipy.signal. This function efficiently finds the indices of all the local maxima (the peaks) in the smoothed signal. The number of detected peaks is the predicted repetition count.

4. **Benchmarking and Evaluation**:

![image](https://github.com/user-attachments/assets/93e67964-93eb-481f-bd9b-6e104f2395ac)

    - A "ground truth" DataFrame is created that contains the known number of repetitions for each set (5 for heavy, 10 for medium).
    - The count_reps function is run on every set in the dataset.
    - The predicted rep counts are compared against the ground truth using Mean Absolute Error (MAE) as the evaluation metric.

5. **Results**: The final algorithm achieved an MAE of approximately 1.0, indicating that, on average, the predicted repetition count is only off by one rep per set—a highly accurate result for a heuristic-based approach.

## 📈 Final Results & Conclusion
This project successfully demonstrates the construction of a high-performance activity recognition and repetition counting system from raw sensor data.

1. **Classification Accuracy**: The final Random Forest model achieved a remarkable 99.4% accuracy on a held-out test set, confirming its robustness and ability to generalize to new users.
  
2. **Repetition Counting Accuracy**: The custom peak-detection algorithm achieved a Mean Absolute Error (MAE) of ~1.0, proving its effectiveness for practical use.

The key takeaway is that with careful data processing, advanced feature engineering (especially in the frequency domain), and rigorous model selection, it is possible to build highly accurate models for complex human activity recognition tasks. This project provides a solid blueprint for developing similar systems for fitness tracking and other Quantified Self applications.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. If you have any ideas for improvement, please open an issue to discuss what you would like to change or submit a pull request. Please adhere to the existing code style and project structure.
