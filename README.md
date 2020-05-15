# EECS6895---project


# Abstract
   Healthcare is a major industry in the United States and it is important in the lives of many citizens, but unfortunately the continuingly rising in medicare care overall experienture and high costs of health-related services leave many patients with limited medical care.  
     In order to reduce costs for either insurance companies or the government, our project aims to build a clinical retrieval system to detect anomaly cases among electronic medical records. Previous work has shown that Machine learning for data-driven diagnosis has been actively studied in medicine to provide better healthcare and publicly available Medicare claims data can be leveraged to construct models capable of automating fraud detection.However, the challenges associated with characstic of electronic clinical records may include: imbalanced-class big data, irregularity in time, and sparsity hinder performance.  
     To address the challenge, we first used a method to calculate similarities between medical records by dealing with these medical records as event and sequence embeddings based on existing literature. To make a comparison of sequences with different lengths easier, our system incorporates Dynamic time warping sequence based alignment.  
     We then implemented two anomaly detection approaches among patients events sequence data: Dynamic time warping (DTW) and LSTM-based Variational AutoEncoder (VAE). They are then followed by Local Outlier Factor (LOF) and other anomaly detection analysis separately. At the final stage, we develop a visual analytics system to support comparative studies of patient records.Through its interactive interface, the user can quickly identify patients of interest and conveniently review both the temporal and multivariate aspects of the patient records. 
  
# Datasets
  In this project, we used detailed and comprehensive patients’ data from MIMIC-III dataset which contains 26 tables and medical records for 41000+ critical care patients. MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising de identified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.It includes demographics, vital signs, laboratory tests, medications, and more. Details are available on the MIMIC website: https://mimic.physionet.org/

# Contact
If you have any question please contact Chengming Xie cx2234@columbia.edu or Hongshan Lin hl3353@columbia.edu
