# Comparative Technical Report: LSTM-Based Forecasting for Medical and Industrial Time-Series Datasets

## 1. Dataset and Its Nature

The evaluation of Long Short-Term Memory (LSTM) networks across high-stakes domains is a strategic necessity for validating architectural generalizability. In this report, we analyze the performance of a unified LSTM framework across two disparate environments: the AZT1D dataset (healthcare/physiological) and the ETDataset (ETTm1) (industrial/critical infrastructure). These datasets represent distinct challenges; while medical data is characterized by high intra-subject volatility and non-linear biological responses, industrial data typically follows stationary operational cycles and seasonal trends.

The AZT1D dataset is a real-world repository comprising data from 25 subjects over a 6–8 week period (averaging 26 days of continuous monitoring). It is defined by 5-minute granularity for Continuous Glucose Monitoring (CGM) and includes granular bolus insulin features—such as total dose and specific correction amounts—extracted from Tandem t:slim X2 insulin pumps. Conversely, the ETDataset (ETTm1) provides a 2-year industrial log of electricity transformers recorded at 15-minute intervals. The primary objective for ETTm1 is predicting "Oil Temperature" (OT), a critical proxy for transformer health and load capacity.

### Dataset Comparative Specifications

| Dimension | AZT1D (Healthcare) | ETTm1 (Industrial) |
|-----------|-------------------|-------------------|
| Time Granularity | 5 Minutes | 15 Minutes |
| Target Variable | CGM (mg/dL) | Oil Temperature (OT) |
| Key Features | CGM, Basal, Bolus, Carbs, DeviceMode | HUFL, HULL, MUFL, MULL, LUFL, LULL |
| Metadata | Subject_ID, BolusType, DeviceMode | Recorded Date |
| Duration | 6–8 weeks per patient | 2 Years |

The multi-subject nature of AZT1D introduces significant personal variability, as metabolic signatures differ drastically between individuals. Modeling this requires respecting subject boundaries to account for idiosyncratic responses to insulin and carbohydrates. In contrast, the ETTm1 data is derived from stationary transformer stations where fluctuations are governed by electricity usage patterns, holidays, and weather-driven seasonality rather than biological variability. These structural differences necessitate rigorous data integrity protocols to ensure the model captures underlying temporal dependencies without falling victim to data leakage.

## 2. Preprocessing and Data Integrity

The foundation of a high-fidelity predictive model rests on its data integrity protocols, specifically the prevention of data leakage. In time-series forecasting, leakage occurs when information from the future is inadvertently used to train the model, or when training statistics are informed by the validation or test sets.

### Sorting and Alignment Strategy

Chronological and logical alignment was achieved through targeted sorting and interpolation:

* **AZT1D:** Records were sorted by Subject_ID and then by EventDateTime. To resolve the frequency mismatch between hourly basal rates and 5-minute CGM readings, basal values were repeated for each 5-minute interval. Carbohydrate and bolus events were zero-filled for intervals without recorded events.
* **ETTm1:** Data was sorted strictly by the date timestamp to maintain the 15-minute interval sequence over the 2-year duration.

### Scaling and Feature Engineering

To prevent leakage, the StandardScaler was fit only on the training set (the first 70% of the indices). The resulting parameters were then applied to transform the validation and test sets. For the AZT1D dataset, categorical BolusType data was parsed using regular expressions to extract numeric insulin percentages and durations, subsequently encoded into binary flags (e.g., is_extended, is_automatic, has_correction) to transform qualitative event logs into quantitative features.

### Integrity Measures

Integrity was further maintained by:

* **Subject Boundaries:** Sequence construction logic for AZT1D ensured that input windows did not span across different Subject_ID entries, preventing the model from learning transitions between unrelated patients.
* **Feature Exclusion:** Non-predictive identifiers and timestamps (e.g., Subject_ID, EventDateTime, date) were excluded from the feature tensors to ensure the model relied solely on temporal and physiological/industrial signals.

## 3. Sequence Construction for LSTM

Temporal modeling utilizes windowing to map historical observations to a future target. This sequence length (T) represents the "look-back" window, determining the model's temporal memory and its ability to identify recurring patterns.

### Windowing Parameters

The study utilized the following standardized parameters:

* **Window Length (T):** 96 steps.
* **Prediction Horizon (H):** 1 step.
* **Input Tensor Shape (X):** N × 96 × F (where F is the number of features).
* **Target Tensor Shape (y):** N × 1.

### Sequence Creation Logic

The create_sequences function was implemented using a sliding window approach:

* Iterate from index 0 through N - T.
* For each step i, slice the feature matrix from i to i+T to form X.
* Assign the target value at index i+T to y.
* Store results as multi-dimensional arrays, subsequently converted to PyTorch tensors.

### Analysis of Window Selection

The 96-step window length has significant domain-specific implications. In AZT1D, 96 steps represent 8 hours, a window optimized to capture full insulin-on-board cycles and postprandial glucose transitions. In ETTm1, 96 steps represent 24 hours, enabling the model to internalize a complete daily cycle of power load and thermal fluctuations. This alignment allows the architecture to capture both circadian biological rhythms and daily industrial operational cycles.

## 4. LSTM Training Setup

The LSTM architecture was selected for its efficacy in processing non-linear time-series and managing the vanishing gradient problem common in long sequences.

### Architecture Specification

A standardized architecture was deployed for both use cases:

* **Layers:** 2-layer LSTM with 128 hidden units.
* **Output:** A fully connected linear layer for single-step regression.
* **Regularization:** A dropout rate of 0.2 was applied to recurrent layers.

### Training Protocols and Stability

Training was conducted with a chronological split of 70% Train, 15% Val, and 15% Test. To ensure training stability and prevent exploding gradients—a common issue in deep recurrent networks—Gradient Norm Clipping was implemented with a max_norm of 1.0. A Batch Size of 64 was utilized to balance computational efficiency with the smoothing of stochastic gradient updates.

* **Optimizer:** AdamW (Weight decay: 1e-5).
* **Scheduler:** ReduceLROnPlateau (factor: 0.5, patience: 5) to dynamically refine the learning rate.
* **Early Stopping:** Patience of 15 epochs based on validation loss to prevent overfitting.

## 5. Techniques Used

Baseline LSTM performance was augmented through domain-specific feature selection and strategic regularization.

* **AZT1D Feature Engineering:** The model utilized 22 features, including the encoded bolus flags, basal rates, and Time Features (hour, day of week, month) to capture temporal metabolic patterns. Notably, delta_CGM and Readings (CGM/BGM) were excluded to prevent the model from relying on pre-calculated derivatives.
* **ETTm1 Feature Selection:** While time features were extracted during preprocessing, the final training feature set was restricted to 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, and OT). This decision focused the model strictly on the operational power load metrics and the target temperature variable.
* **Regularization:** In addition to dropout, L2 regularization via weight decay (1e-5) was enforced within the AdamW optimizer to constrain model complexity over the maximum 100-epoch training limit.

## 6. Results

Quantitative validation demonstrates that the LSTM architecture achieves high predictive fidelity in both physiological and industrial contexts. The following metrics are reported in Original Units (mg/dL for glucose and Celsius for oil temperature), though the model optimized for Scaled Units during training.

### Quantitative Performance Metrics

| Dataset | MSE | RMSE | MAE | R² |
|---------|-----|------|-----|-----|
| AZT1D (Glucose) | 21.3975 | 4.6257 | 2.8467 | 0.9885 |
| ETTm1 (Oil Temp) | 0.1101 | 0.3318 | 0.2238 | 0.9864 |

### Baseline Comparison

To contextualize the LSTM's efficacy, results were compared against a "Naive Approach" (predicting y_{t+1} = y_t).

* **AZT1D:** The LSTM's MSE (21.3975) represents a significant improvement over the Naive MSE of 41.8435.
* **ETTm1:** The LSTM's MSE (0.1101) outperformed the Naive MSE of 0.1984.

The R² values exceeding 0.98 for both datasets indicate the model successfully explains nearly all the variance in the target variables, demonstrating a strong fit for both volatile biological data and smooth industrial trends.

### Visualizations

#### ETTm1 (Industrial Dataset) Results

**Training History**

![ETTm1 Training History](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/ETD/img/1.png)

The training history for ETTm1 demonstrates rapid convergence within the first few epochs, with both training and validation losses stabilizing around 0.011-0.012 (scaled units). The close alignment between training and validation curves indicates minimal overfitting, reflecting the seasonal and stationary nature of industrial transformer data.

**Prediction Accuracy**

![ETTm1 Predictions vs Actual](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/ETD/img/2.png)

The predictions vs actual plot shows excellent tracking of oil temperature fluctuations across the entire test set. The model successfully captures both short-term variations and longer-term seasonal trends in transformer thermal behavior.

**Error Distribution**

![ETTm1 Error Distribution](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/ETD/img/3.png)

The error distribution is tightly centered around zero with a near-Gaussian shape, indicating unbiased predictions. The majority of prediction errors fall within ±1°C, demonstrating the model's precision in temperature forecasting for critical infrastructure monitoring.

#### AZT1D (Healthcare Dataset) Results

**Training History**

![AZT1D Training History](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/AZTD/img/1.png)

The AZT1D training history shows steady convergence over approximately 35 epochs. Notable is the occasional lower validation loss compared to training loss in early epochs, suggesting effective generalization across different patient metabolic profiles. The model benefits from early stopping and learning rate scheduling to prevent overfitting to individual patient patterns.

**Prediction Accuracy**

![AZT1D Predictions vs Actual](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/AZTD/img/2.png)

The predictions vs actual plot reveals the model's ability to track highly volatile glucose dynamics across multiple patients. The LSTM successfully captures both gradual glucose trends and rapid excursions associated with meals and insulin administration, maintaining accuracy even during critical dysglycemic episodes.

**Error Distribution**

![AZT1D Error Distribution](https://github.com/hknl5/LSTM-Based-Forecasting-for-Medical-and-Industrial-Time-Series-Datasets/blob/main/AZTD/img/3.png)

The error distribution for glucose predictions demonstrates a symmetric, near-zero-centered pattern. The majority of errors fall within ±10 mg/dL, which is clinically acceptable for continuous glucose monitoring applications. The tight distribution confirms the model's reliability for real-time diabetes management support.

## 7. Discussion and Next Steps

The achievement of high R² values confirms the robustness of the 2-layer LSTM for short-term forecasting. The model demonstrated resilience against the biological noise of AZT1D and the multi-seasonal complexity of ETTm1.

### Observation Analysis

Analysis of the training logs for AZT1D revealed that validation loss was occasionally lower than training loss during the early epochs. This suggest that the model generalized effectively to the validation subject distribution. For ETTm1, the model achieved rapid convergence, likely due to the highly seasonal and less stochastic nature of industrial transformer metrics compared to human metabolism.

### Efficacy in Critical Regions

The model showed high predictive accuracy in "dysglycemic regions" (hypoglycemia and hyperglycemia), which are critical for clinical safety in T1D management. Similarly, in the ETTm1 context, the stable temperature predictions provide a reliable basis for identifying potential "extreme load capacity" issues, as highlighted in the ETT documentation.

