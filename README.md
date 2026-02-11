# Comparative Analysis of Deep Learning Architectures for Time-Series Forecasting: AZT1D and Electricity Transformer Datasets

## 1. Abstract

The strategic advancement of predictive modeling in high-stakes domains—specifically healthcare and industrial infrastructure—relies heavily on the availability and utilization of high-quality, real-world datasets. In clinical settings, longitudinal physiological data is the cornerstone of personalized therapy and digital twin systems, while industrial load data is essential for maintaining grid stability and preventing equipment depreciation. 

This study evaluates the performance of supervised forecasting architectures on two distinct benchmarks: the **AZT1D (Type 1 Diabetes)** physiological dataset and the **ETT (Electricity Transformer)** dataset. We employ Long Short-Term Memory (LSTM) and Transformer architectures to predict next-step outcomes based on an 8-hour or 24-hour historical context. 

Empirical results demonstrate that while both models achieve robust performance, the LSTM consistently outperforms the Transformer across all metrics (MSE, MAE, R²). This findings suggest that for short-horizon forecasting (H=1), the sequential inductive bias of recurrent architectures provides superior modeling of local temporal continuity compared to the global receptive field of self-attention mechanisms.

## 2. Datasets and Challenges (AZT1D, ETT/ETDataset)

Dataset diversity is a fundamental requirement for assessing the generalizability and robustness of time-series architectures. Models must be tested against varying temporal structures, from the highly stochastic metabolic fluctuations in human physiology to the rhythmic but irregular load patterns of mechanical systems.

### AZT1D Dataset

The **AZT1D dataset** represents a landmark resource in computational medicine, featuring a 25-individual cohort of patients with Type 1 Diabetes (T1D). Collected over a 6–8 week duration, it captures naturalistic diabetes management through Continuous Glucose Monitoring (CGM) and automated insulin delivery (AID) systems. Unlike simulated datasets, AZT1D includes granular details such as:

- Bolus types (standard, correction, or automatic)
- Specific device modes (e.g., sleep or exercise)

These provide critical context for glycemic excursions.

### ETT Dataset

The **ETT (Electricity Transformer Dataset)** serves as a benchmark for long-sequence forecasting in industrial contexts. It comprises load and Oil Temperature (OT) measurements from transformers over a two-year period. While ETT exhibits strong seasonal trends, it is frequently disrupted by irregular patterns and extreme load events, making accurate forecasting essential for proactive system management and waste reduction.

### Dataset-Specific Challenges

Developing these models requires addressing dataset-specific challenges:

- **For AZT1D**: A primary hurdle involves the multi-modal nature of data storage; while CGM and bolus logs are typically CSV-based, basal rates and device modes often require Optical Character Recognition (OCR) extraction from Tandem pump PDF reports.

- **For ETT**: The challenge lies in capturing the interplay between local continuity and long-term seasonal cycles.

### Table 1: Dataset Comparison

| Feature | AZT1D | ETT / ETDataset (m1) |
|---------|-------|---------------------|
| **Target Variable** | Continuous Glucose Monitoring (CGM) | Oil Temperature (OT) |
| **Feature Set** | Basal, Bolus (Total/Type/Correction), Carbs, Device Mode | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| **Granularity** | 5-minute intervals | 15-minute intervals (4x per hour) |
| **Structure** | Multi-subject physiological cohort | Multi-station industrial sensor logs |
| **Primary Challenges** | OCR extraction from PDFs; Multi-subject boundaries | Seasonal/Irregular pattern mix; equipment depreciation risks |

These distinct data structures are formally defined to facilitate supervised machine learning tasks.

## 3. Problem Formulation

Time-series forecasting is formulated here as a supervised learning task where a model learns a mapping from a fixed historical window to a single-step future target. In clinical and industrial operations, this proactive modeling allows for early intervention, such as preventing hypoglycemia or managing transformer overheating.

The mathematical framework utilizes a **sliding window approach** with a context size (T=96) and a forecast horizon (H=1):

- **For AZT1D**: 96 steps represent 8 hours of history used to predict CGM at t+1
- **For ETTm1**: The same window represents 24 hours of data

The primary target variables are:
- Continuous Glucose Monitoring (CGM)
- Oil Temperature (OT)

Although the raw AZT1D logs contain a `delta_CGM` feature, it was ignored in favor of predicting absolute CGM values to ensure a consistent forecasting objective across datasets.

## 4. Preprocessing and Leakage Prevention

Rigorous preprocessing is essential to maintain data integrity and prevent temporal leakage, which occurs when information from the validation or test sets "leaks" into the training phase.

### Chronological Splitting Strategy

We adopted a chronological splitting strategy:
- **70%** for training
- **15%** for validation
- **15%** for testing

To ensure strict methodological transparency, the **StandardScaler** was fit exclusively on the training portion (e.g., indices 0 to 214793 for AZT1D). The derived parameters were then applied to scale the validation and test sets.

### Multi-Subject Boundary Management

A critical methodological concern in the AZT1D dataset is the management of multi-subject boundaries. The data was sorted by `Subject_ID` and `EventDateTime` before sequence creation. Because the standard `create_sequences` function was applied globally, "bridge" sequences are generated at the transition point between two subjects (e.g., a window composed of the final 95 steps of Subject 1 and the first step of Subject 2). 

In high-precision clinical modeling, these sequences represent a data integrity risk as they treat unrelated physiological states as a continuous stream.

### Feature Encoding

- Categorical features such as `DeviceMode` and `BolusType` were encoded into binary flags (e.g., `is_extended`, `is_sleep`) or numeric representations
- For AZT1D, hourly basal rates were forward-filled to align with the 5-minute CGM readings

## 5. Sequence Construction

Sliding window algorithms transform raw tabular data into 3D tensors suitable for deep learning. This process generates:
- An input tensor **X** with shape `[n_sequences, T, D]`
- A target vector **y** with shape `[n_sequences, 1]`

The input dimension **D** is dataset-dependent:

- **AZT1D (D=22)**: Encoded features include glucose, carbs, bolus metrics, device modes, and time features
- **ETTm1 (D=7)**: Includes six load features and the target oil temperature

These sequences enable the models to learn the temporal dependencies inherent in the data.

## 6. Models

This analysis evaluates the theoretical and empirical trade-offs between Recurrent Neural Networks (RNNs) and Self-Attention mechanisms.

### LSTM Architecture

The Long Short-Term Memory (LSTM) architecture utilized two hidden layers with 128 units each. The model leverages its internal gating mechanism to maintain a cell state across the T=96 window. 

**Key specifications:**
- Two hidden layers with 128 units each
- Dropout rate: 0.2
- Final prediction produced by mapping the hidden state from the terminal time-step of the second layer to the output dimension via a linear fully connected layer

### Transformer Architecture

The Transformer model employs a `TransformerEncoderLayer` with 8 attention heads. Positional encoding is added to the input projection to preserve temporal order. 

**Architectural specifications differed by dataset complexity:**

**AZT1D Transformer:**
- Leaner configuration with `d_model=64` and `dim_feedforward=128`
- Higher dropout rate (0.5) due to higher noise in physiological data

**ETT Transformer:**
- Larger `d_model=128` to capture complex industrial load patterns

The Transformer processes the entire window in parallel, extracting the final prediction from the last time-step's representation in the encoder stack.

## 7. Experimental Design

Standardized experimental configurations were maintained to ensure reproducibility. Training utilized the AdamW optimizer and the Mean Squared Error (MSE) loss function.

### Table 2: Experiment Configuration Summary

| Parameter | Configuration Value |
|-----------|-------------------|
| **Window (T) / Horizon (H)** | 96 / 1 |
| **Data Splits (Tr/V/Te)** | 70% / 15% / 15% |
| **Scaling Strategy** | StandardScaler (Fit on Training Only) |
| **Optimizer / Learning Rate** | AdamW / 0.001 |
| **Weight Decay** | 1e-5 |
| **Gradient Clipping** | max_norm=1.0 |
| **Batch Size** | 64 |
| **Early Stopping Patience** | 15 |
| **Evaluation Metrics** | MSE, MAE, R² |

## 8. Results and Analysis

Empirical results across both datasets reveal a consistent performance advantage for the LSTM over the Transformer.

### AZT1D Performance 

**LSTM:**
- MSE: 21.3975
- MAE: 2.8467
- R²: 0.9885

**Transformer:**
- MSE: 36.3333
- MAE: 4.0278
- R²: 0.9804

### ETTm1 Performance

**LSTM:**
- MSE: 0.1101
- MAE: 0.2238
- R²: 0.9864

**Transformer:**
- MSE: 0.1501
- MAE: 0.2796
- R²: 0.9814

### Methodological Synthesis

The superior performance of the LSTM can be attributed to the **sequential inductive bias** of recurrent architectures. For a short-step horizon (H=1), the primary predictive signal is the local temporal continuity of the signal. 

LSTMs are inherently biased toward this sequential order, whereas Transformers use global self-attention to weight the entire window. In the H=1 setup, the global receptive field provides marginal utility, and the lack of an inherent sequential bias in Transformers—coupled with higher regularization requirements (0.5 dropout in AZT1D)—leads to higher error rates. 

Early stopping occurred at:
- **Epoch 33** for the AZT1D LSTM
- **Epoch 36** for the ETT Transformer

This indicates sufficient convergence.

## 9. Limitations

Transparency regarding the boundaries of this study is essential for clinical and industrial deployment.

### Sample Size and Generalizability

The AZT1D cohort is limited to 25 individuals; while granular, it lacks the scale to represent the full heterogeneity of the T1D population.

### Data Extraction Concerns

The reliance on OCR for extracting basal rates from PDF reports introduces a potential source of noise.

### Forecasting Horizon

Finally, the study is restricted to a single-step horizon (H=1). In real-world AID systems, a multi-step horizon (e.g., 30–60 minutes) is required for safe insulin dosing, an area where the Transformer's global attention may eventually provide more significant advantages.

---

Chao
