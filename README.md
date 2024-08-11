# Naples Diaper Market Geo-Analytics for Potential Estimation

## Project Overview
Assessing the diaper market potential for Fater (a P&G subsidiary) and refining revenue forecasts for Naples stores through socio-demographic, geographic, and points of interest analysis.

This project explores the estimation of potential revenue for Naples stores throughout the entire province, with a focus on the Italian diaper market (in 1000s). The study incorporates a detailed approach, considering socio-demographic factors, geographical features, and points of interest, to extract insights for strategic decision-making in the competitive retail landscape.

A primary revelation from our analysis is the influence of hypermarkets and larger stores in shaping potential revenue dynamics. These establishments demonstrate a considerable capacity for customer retention and sales generation. Their expansive footprint in the market positions them as pivotal players, warranting strategic marketing emphasis to leverage their inherent potential fully.

A critical determinant identified in our study is the role of parking availability, showcasing an impact on potential values. Stores equipped with parking facilities exhibit higher potential value, underscoring the significance of convenient access for customers. This insight enables businesses to tailor promotional strategies, highlighting the presence of parking facilities as a compelling feature for attracting and retaining customers.

Temporal dynamics form a key facet of our findings, with distinct gravitational patterns emerging during weekends and specific evening time slots, particularly from 17:00 to 20:00. This temporal analysis provides businesses with a strategic edge, allowing them to optimize promotional campaigns during peak hours, aligning with heightened customer gravitation.

Delving into demographics, our study identifies specific age groups displaying a higher inclination towards stores, notably individuals aged 41-50 and those over 60. Additionally, a gender-based trend surfaces, with males exhibiting a higher average gravitation towards stores.

In summary, the FATER Geo-Analytics Challenge delivers actionable recommendations for strategic marketing focus, targeted promotions, and an emphasis on parking facilities. This comprehensive roadmap equips businesses to navigate the landscape of the Italian diaper market, unlocking the full potential for revenue maximization.

### Analysis Steps
- Exploring Data Types, Dimensions, and Spatial Relationships
- Performing spatial exploration and analysis
- Analyzing Categorical and Numerical Features' Scales
- Handling missing values
- Exploring Descriptive Statistics
- Univariate Analysis: Visualizing Numerical Features
- Univariate Analysis: Visualizing Categorical Features
- Bivariate Analysis: Exploring Relationships Between Numerical Predictors and Potential
- Bivariate Analysis: Exploring Relationships Between Categorical Predictors and Potential
- Hypothesis Testing: Assessing Statistical Significance of features using Kruskal-Wallis and Mann-Whitney U tests
- Correlation Analysis: Examining Relationships Between Numerical Predictors and Potential
- Contingency Table of Store Potential analyses with respect to store type, store size, and parking status.
- Average population gravitation analyses with respect to day type, time slot, and demographics.
- Geo-Spatial analysis: Visualizing Potential Across Different Store Types
- Pareto Analysis: Identifying stores and store types that generate the most potential

### Modeling
- The hyperparameter tuning process involved exploring various configurations for the Decision Tree, Random Forest, Gradient Boosting, and AdaBoost models using 80% of the data for training while reserving 20% as a hold-out set—data that the model has not seen. Bagging was used to reduce variance (overfitting) while boosting was employed to reduce bias (underfitting).
- These model parameters were subjected to k-fold cross-validation (k = 10) to identify the best-performing configurations. The choice of k = 10 aligns with established best practices, supported by empirical evidence. Ron Kohavi's experiments on diverse real-world datasets indicate that a 10-fold cross-validation strategy strikes an optimal balance between bias and variance in model assessment.
- Following the hyperparameter tuning, the models underwent rigorous evaluation using a 10-fold cross-validation approach. The root mean squared error (RMSE) was employed as a key metric to assess the predictive performance of each model. The RMSE values give an indication of how well each model is performing in terms of predicting the target variable. The lower the RMSE, the better the model's predictions align with the actual values.
- The optimal model was then trained on the entire 100% of the training data. Subsequently, the model underwent evaluation on the reserved holdout set to assess its performance. Mean Squared Error (MSE), Mean Absolute Error (MAE), and coefficient of determination (R2) metrics were employed to evaluate the Random Forest Regressor's performance.

## Usage
### Prerequisites:
- pandas
- numpy
- seaborn
- matplotlib
- statsmodels.api
- scipy.stats
- matplotlib.colors
- geopandas
- shapely
- sklearn

## License:
This project is licensed under the Raza Mehar License. See the LICENSE.md file for details.

## Contact:
For any questions or clarifications, please contact Raza Mehar at [raza.mehar@gmail.com], Pujan Thapa at [iampujan@outlook.com] or Syed Najam Mehdi at [najam.electrical.ned@gmail.com].

## Bibliography & References
- Alan Agresti, Maria Kateri. “Foundations of Statistics for Data Scientists with R and Python”. CRC Press. 2022.
- Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili. “Machine Learning with PyTorch and Scikit-Learn”. Packt Publishing Ltd. 2022.
- Aurélien Géron. “Hands-On Machine Learning with Scikit-Learn & TensorFlow”. O’Reille Media. 2019.
