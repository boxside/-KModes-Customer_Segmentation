# [KModes]Customer_Segmentation

 ## About
 ### Code and Resources Used 
  **Python Version:** 3.8  
  
  **Packages:** pandas, KModes, matplotlib, seaborn, sklearn, pickles

  **Dataset:**  DQLab
 
 ### list data details : 
  * Customer ID: with format mixed text CUST- then number
  * Nama Pelanggan: Customer name
  * Jenis Kelamin: Gender
  * Umur: Age
  * Profesi: Job/profession
  * Tipe Residen: Resident type ,2 categories: Cluster dan Sector.
  * Nilai Belanja Setahun: Expense value per year
  
## Data Handling
* Library preparation and read dataframe
* Missing Value checking
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/dataprep.png)
* result : there are 50 rows and 7 columns (2 numerical columns & 5 categorical columns),No Missing value
## Exploratory Data Analysis
### Numerical variabel EDA
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/index.png)
### Categorical variabel EDA
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/kolom_kategorikal1.png)
### Result
 * Average of Customer : 37.5 year
 * Average from Customer expense value per year : 7,069,874.82
 * Customer dominated with Female which  41 people (82%) and Male which 9 people (18%)
 * most customer's proffesion : Wiraswasta(entrepreneur) (40%) and followed by Professional (36%) and other (24%)
 * from all customer around 64% living in cluster resident and 36% living in sector resident
## Modelling
* Standarized features's value
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/standarized.png)
* converting features using labelencoder
* find Optimal cluster value
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/k-cluster.png)
  the optimal value is 5
* Modelling using KModes
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustering.png)
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustervsnilai.png)
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustervsres.png)
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustervsprof.png)
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustervsumur.png)
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/clustervsjk.png)
* Segment naming 
![Figure 1](https://github.com/boxside/-KModes-Customer_Segmentation/blob/main/figure/segmentation.png)
  * Cluster 0: Diamond Young Entrepreneur, Average expense value per year around 10M. with cust.ages range  18 - 41 year with average : 29 tahun.
  * Cluster 1: Diamond Senior Entrepreneur, Average expense value per year around 10M. with cust.ages range  45 - 64 year with average : 55 tahun.
  * Cluster 2: Silver Students, for students with average of their age : 16 year and Average expense value per year around 3M.
  * Cluster 3: Gold Young Member, for young professional and young housewife with range 20 - 40 year with average : 30 year and Average expense value per year around 6M.
  * Cluster 4: Gold Senior Member, for senior professional and senior housewife with range  46 - 63 year with average : 53 year and Average expense value per year around 6M.
