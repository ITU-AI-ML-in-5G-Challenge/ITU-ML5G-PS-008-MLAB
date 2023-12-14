# ITU-ML5G-PS-008-Network-failure-classification-model-using-NDT--MLAB

This code is used for MLP-base feature selection in KDDI challenge. We mainly operate the code in Google colabortory. Therefore, the dataset and model import are all related to the google drive.

Given the randomness of neural network, the code might not produce the exact same result as that in the slides

Also, the code still need further refinement.

# Repository structure
data: folder containing the datasets for domain A and domain C
MLP_based_feature_selection: The entire python code to address the NDT problem

# Proposed model
We used multi-layer perceptron to train the model on domain A. After that, we applied transfer learning by freezing all the layer but the last layer. We fine-tuned the last layer with training data from domain C

# Feature selection
The originality of the research mainly lies on feature selection process. We calculate the mutual information betwwen the label column and feature columns and select different numbers of top features. By changeing the value K, we can evaluate the performance when we use different number of selected features.
'''
t0 = time.time()
k_best_A = SelectKBest(score_func=mutual_info_classif, k=150)
X_new_A = k_best_A.fit_transform(data_input_features_A, encoded_label_A)
'''
