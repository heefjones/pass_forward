![](./images/slide_1.PNG)

# Goal
How can we measure QB performance? Passing statistics like yards, touchdowns, or completion % are often not indicative of a player's performance. Rating systems like NFL passer rating or ESPN's QBR also have their weaknesses and can be heavily influenced by other players on the field. [Pro Football Focus](https://www.pff.com/)'s offensive grade is an aggregate measure of a quarterback's overall performance. The goal of this project is to create a predictive model that takes in a quarterback's past statistics to predict their __PFF offensive grade__ in the next season. 

## Data
The data used in this project was completely sourced from [PFF's website](https://www.pff.com/). 
- 1398 QB seasons
- 299 unique NFL QBs
- 19 NFL seasons (2006-2024)

## Modeling
### XGBoost
- Used last season's stats as well as mean and std of all stats over 5 years, and mean/std stats over the entire career.
- 5-split KFold cross validation.
- validation RMSE of 12.22.

### RNN
- Used the entire sequence of QB's past data.
- 80/20 train/val split.
- validation RMSE of 14.74.

Due to the noticeably weaker performance of the RNN, I only proceeded with the XGBoost.

## 2024 Predictions
Next, the XGBoost was trained on the 2006-2022 data, and used the 2023 data to predict 2024 grades (as a holdout test set).
![](./images/xgboost_2024_preds.PNG)
Each point represents a single player in 2024. The distance from the black line is how far off our prediction was. A perfect model would only have dots on the line. The final RMSE on the 2024 holdout set was __11.49__ with an R^2 of __0.36__.

## 2025 Predictions
Finally, the model was trained on all data (2006-2023) and used the 2024 data to make 2025 predictions. Here are the final results:
![](./images/xgboost_2025_preds.PNG)

## Files
─ eda.ipynb - Data, combining, EDA, and feature engineering.
─ xgboost.ipynb - Bayesian optimize XGBoost model and predict on both 2024 and 2025 seasons.
─ rnn.ipynb - Train and validate a RNN.
- helper.py - Custom functions for data processing, visualization, and model training.
─ presentation.pdf - Slide deck to summarize findings.

## Repository Structure
```
├── eda.ipynb
├── xgboost.ipynb
├── rnn.ipynb
├── helper.py
├── README.md
├── presentation.pdf
├── .gitignore
└── /images
```
