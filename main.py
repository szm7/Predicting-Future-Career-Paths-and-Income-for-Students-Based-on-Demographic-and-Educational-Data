
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Title
st.title("Predicting Future Career Paths and Income for Students Based on Demographic and Educational Data")

# Problem Statement
st.divider()
st.header("Problem Statement")
st.write("""As students progress through their journey, many face uncertainty about their future career paths and potential income. This uncertainty can lead to mediocre educational and career choices, affecting their long-term success and financial stability. There is a growing need for a data-driven approach to guide students in making informed decisions about their education and career paths.""")


# Objective
st.divider()
st.header("Objective")
st.write(""" The objective of this project is to develop a predictive model that can forecast student's future career paths and potential income based on demographic and educational data. The model will analyze various factors, such as academic performance, socio-economic background, and extracurricular involvement, to predict the most likely career path and associated income levels. The insights generated from this model will assist students, educators, and policymakers in making data-driven decisions to optimize educational and career outcomes.""")

# Dataset
st.divider()
st.header("Dataset Overview")
st.write("""
In this section, we will explore the synthetic dataset used for this analysis. The dataset is central to our project, as it provides the necessary information to predict student success.
The dataset used for this project was a synthetic dataset made by merging the necessary columns from 2 different datasets related to student demographics and future job and income data. 
""")
column_explanations = """
- **Base Pay:** The annual base salary of the individual.
- **Position:** The job title or role held by the individual.
- **sex:** The gender of the individual (e.g., Male, Female).
- **age:** The age of the individual in years.
- **Medu:** The education level of the mother, on a scale (e.g., 0: None, 1: Primary, 2: Secondary).
- **Fedu:** The education level of the father, on a scale (e.g., 0: None, 1: Primary, 2: Secondary).
- **Mjob:** The occupation of the mother (e.g., Public, Teacher).
- **Fjob:** The occupation of the father (e.g., Public, Health).
- **studytime:** The weekly study time of the individual, measured in hours.
- **failures:** The number of past academic failures.
- **schoolsup:** Whether the individual received additional educational support (Yes/No).
- **higher:** Whether the individual plans to pursue higher education (Yes/No).
- **internet:** Whether the individual has internet access at home (Yes/No).
- **absences:** The number of school days missed by the individual.
- **SSC:** The mark of the individual in 10th.
- **HSC:** The mark of the individual in 12th.
- **Grad:** The final grade of the individual.
"""
st.markdown(column_explanations)
import pandas as pd

df = pd.read_csv(r"C:\Users\muzza\Downloads\new_modified_dataset.csv")
# Display the dataframe in Streamlit
st.dataframe(df, use_container_width=True)

columns_to_scale = ['Grad', 'HSC', 'SSC']

# Divide each column by 2 to scale from 20 to 10
df[columns_to_scale] = df[columns_to_scale] / 2

# Optional: Round the values to eliminate any floating-point issues
df[columns_to_scale] = df[columns_to_scale].round(2)

# EDA section
st.divider()
st.header("EDA")

st.divider()
st.header("Scatter Plot of SSC vs. HSC by Job Role")

# Scatter Plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x=df['SSC'], y=df['HSC'],
                     c=pd.Categorical(df['Job Role']).codes,
                     cmap='viridis',
                     s=32, alpha=.8)

# Add a color bar to show the mapping of colors to job roles
colorbar = plt.colorbar(scatter, ax=ax, label='Job Role')

# Customize plot appearance
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('SSC')
ax.set_ylabel('HSC')
ax.set_title('Scatter Plot of SSC vs. HSC by Job Role')

# Display the plot in Streamlit
st.pyplot(fig)

st.divider()
st.header("Distribution of Base Pay")

# Create the histogram plot
fig, ax = plt.subplots()
df['Base Pay'].hist(bins=20, edgecolor='black', ax=ax)
ax.set_title('Distribution of Base Pay')
ax.set_xlabel('Base Pay')
ax.set_ylabel('Frequency')

st.write("This bar chart graph shows us the distribution of base pay or income for the students in the future. The graph indicates that the most of the students ended up having a base pay in the range of 40k-50k. The least number of students were in range of 90k and above.")

# Display the plot in Streamlit
st.pyplot(fig)

st.divider()
st.header("Base Pay Distribution by Job Role")

# Create the box plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Job Role', y='Base Pay', palette='viridis', ax=ax)
ax.set_title('Base Pay Distribution by Job Role')
st.write("This bar chart graph shows us the distribution of base pay based on the job role. The graph indicates the range of base pay with respect to their job roles.")

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("HSC Distribution by Job Role")
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x='Job Role', y='HSC', palette='viridis', ax=ax)
ax.set_title('HSC Distribution by Job Role')
st.write("This visualization helps understand the relationship between high school performance and eventual job role, suggesting that certain roles may have higher academic prerequisites or attract students with higher academic performance.")
st.pyplot(fig)

# Plot 2: Pairplot - Pairwise Relationships
st.subheader("Pairwise Relationships")
fig = sns.pairplot(df[['SSC', 'HSC', 'Base Pay']], diag_kind='kde')
st.write("This graph shows us the pairwise relationship between SSC, HSC, and Base pay.")
st.pyplot(fig)

# Plot 3: Bar Plot - Average Base Pay by Job Role
st.subheader("Average Base Pay by Job Role")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='Job Role', y='Base Pay', palette='viridis', ax=ax)
ax.set_title('Average Base Pay by Job Role')
st.write("This bar chart shows us the average Base pay for each job role.")
st.pyplot(fig)

# Plot 4: Bar Plot - Class Failures vs. Study Time
st.subheader("Class Failures vs. Study Time")
study_time_groups = df.groupby('studytime')['backlogs'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(study_time_groups.index, study_time_groups.values)
ax.set_xlabel('Study Time')
ax.set_ylabel('Average Class Failures')
ax.set_title('Class Failures vs. Study Time')
st.write("This graph shows us the relationship between the study time of students and the average class failures. It tells us that with increase in the study time, the average of class failures decreases exponentially ")

st.pyplot(fig)

# Plot 6: Heatmap - Correlation Heatmap
st.subheader("Correlation Heatmap")
numerical_df = df.select_dtypes(include=np.number)
corr_matrix = numerical_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Correlation Heatmap')
st.write("This is the correlation graph which shows us the correlation between each of the columns. From the graph we can cleary see that therre is significant correlation between the absences and backlogs with the marks of the students.")

st.pyplot(fig)

# Plot 7: Count Plot - Backlogs by Sex
st.subheader("Backlogs by Sex")
fig, ax = plt.subplots()
sns.countplot(data=df, x="backlogs", hue="sex", ax=ax)
st.write("This graph shows us the number of students who have backlogs by their gender. Not suprisingly, males were leading in this criteria with more number of backlogs.")

st.pyplot(fig)

# Plot 8: Count Plot - Sex Distribution
st.subheader("Sex Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="sex", hue="sex", ax=ax)
st.write("This graph shows us the distribution of genders.")

st.pyplot(fig)

# Plot 9: Count Plot - Tuition by Sex
st.subheader("Tuition by Sex")
fig, ax = plt.subplots()
sns.countplot(data=df, x="tuition", hue="sex", ax=ax)
st.write("This graph shows us the students who attended extra tuition by their gender. More males were seen to attend tuitions")

st.pyplot(fig)

# Plot 10: Count Plot - Pursue Higher Studies by Sex
st.subheader("Pursue Higher Studies by Sex")
fig, ax = plt.subplots()
sns.countplot(data=df, x="pursue_higher_studies", hue="sex", ax=ax)
st.write("This graph shows us the students who pursued higher education by their gender. Both the genders were almost equal in number for pursuing higher education. While females mostly tended to pursue higher education than males.")

st.pyplot(fig)

# Plot 11: Count Plot - Study Time by Sex
st.subheader("Study Time by Sex")
fig, ax = plt.subplots()
sns.countplot(data=df, x="studytime", hue="sex", ax=ax)
st.write("This graph shows us the study time by both the genders. The plot suggests that males tend to spend more time studying in higher time categories (3 and 4) compared to females, who are more prevalent in lower study time categories (1 and 2).")

st.pyplot(fig)


# Plot 13: Count Plot - Job Role by Skills
st.subheader("Job Role by Skills")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Job Role", hue="Skills", ax=ax)
st.write("This graph shows us shows a bar chart depicting the count of individuals in various job roles, each associated with a specific set of skills. The chart shows that there is a higher count of individuals in roles like Engineer, Manager, and Doctor, which typically require a specialized skill set. Roles such as Clerk and Technician are fewer, indicating fewer individuals pursuing these roles within the dataset.")

st.pyplot(fig)


st.divider()
st.header("PREDICTING JOB ROLE BASED ON STUDENT DEMOGRAPHICS")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Job Role'] = le.fit_transform(df['Job Role'])

label_encoder = LabelEncoder()

for column in df.columns:
  if df[column].dtype == 'object':  # Check if the column is categorical
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop(columns=['Job Role'])
y = df['Job Role']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the model
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)


st.code("""
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the model
rf_classifier.fit(X_train, y_train)
""")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.text(report)

# Confusion matrix
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
st.text(conf_matrix)

st.write("Model Inference")
st.write("Accuracy: The model achieved 100% accuracy, meaning that it correctly classified all the samples in the test set.")
st.write("The model has perfectly classified all instances in the test set, which is reflected in all metrics being 1.00. This suggests that the model is extremely well-fitted to the data, but it also raises the possibility of overfitting if the test set isn't representative of unseen data.")
st.write("Precision: For each class, the precision is 1.00, indicating that when the model predicts a given class, it is always correct. Precision is calculated as the number of true positives divided by the sum of true positives and false positives. Recall: For each class, the recall is also 1.00, meaning that the model correctly identified all instances of that class. Recall is calculated as the number of true positives divided by the sum of true positives and false negatives. F1-Score: The F1-score, which is the harmonic mean of precision and recall, is also 1.00 for all classes. This indicates a perfect balance between precision and recall.")
st.divider()
st.header("PREDICTING THE MONTHLY INCOME BASED ON VARIOUS STUDENT DEMOGRAPHIES")

X1 = df.drop(columns=['Base Pay Range'])
y1 = df['Base Pay Range']

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the model
rf_classifier.fit(X_train1, y_train1)

st.code("""
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the model
rf_classifier.fit(X_train1, y_train1)
""")

y_pred1 = rf_classifier.predict(X_test1)

# Evaluate the model

st.subheader("Classification Report")
report1 = classification_report(y_test1, y_pred1)
st.text(report1)

# Confusion matrix
st.subheader("Confusion Matrix")
conf_matrix1 = confusion_matrix(y_test1, y_pred1)
st.text(conf_matrix1)

st.write("Model Inference")
st.write("Accuracy: The accuracy is 100%, which means that the model has correctly classified all the samples in the test set into the correct income categories.")
st.write("The model's performance on this test set is perfect, with all metrics at 1.00. This suggests that the model is extremely well-fitted to the data.")
st.write("Averages: Macro Avg: The macro average for precision, recall, and F1-score is 1.00, showing that the model's performance is uniformly excellent across all income categories. Weighted Avg: Since all classes have been perfectly predicted, the weighted averages for precision, recall, and F1-score are also 1.00.")

st.divider()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.header("PREDICTING GRADUATION MARKS")

st.code("""
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train2, y_train2)
""")

# Display correlation matrix as a heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix[['Grad']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap with Grad')
st.pyplot(plt.gcf())

# Define features and target
X2 = df[['backlogs', 'pursue_higher_studies', 'absences', 'SSC', 'HSC']]
y2 = df['Grad']

# Split the data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train2, y_train2)

# Make predictions
y_pred2 = model.predict(X_test2)

# Evaluate the model
mse = mean_squared_error(y_test2, y_pred2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test2, y_pred2)

# Display the results
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X2.columns, columns=['Coefficient'])
st.write("Model Coefficients:")
st.write(coefficients)

st.write("Model Inference")
st.write("Mean Squared Error (MSE): 0.24. Interpretation: The MSE is a measure of the average squared difference between the actual and predicted values. An MSE of 0.24 suggests that, on average, the squared difference between the predicted and actual graduation marks is 0.24. Since MSE values are in the same units as the squared output variable, this indicates relatively low error.")
st.write("Root Mean Squared Error (RMSE): 0.49. Interpretation: RMSE is the square root of MSE and provides an error metric in the same units as the predicted variable, in this case, graduation marks. An RMSE of 0.51 suggests that the model's predictions are off by about 0.49 marks, on average.")
st.write("R-squared: 0.74. Interpretation: R-squared (R²) is a measure of how well the independent variables explain the variance in the dependent variable (graduation marks). An R² value of 0.74 means that 74% of the variance in graduation marks can be explained by the features used in the model. This indicates a good level of predictive power, but there is still 28% of the variance unexplained, suggesting that there might be other factors influencing graduation marks that aren't captured by the current model.")    
st.write("The model performs reasonably well, with an R² of 0.74 indicating a good fit. The RMSE of 0.49 suggests that the model's predictions are fairly close to the actual values.")

st.header("INFERENCE")
st.write("Students' academic performance (SSC, HSC, Grad) and study habits (studytime, absences, backlogs) significantly influence their future career paths and potential income. Students with higher academic performance tend to pursue higher-paying job roles and have better career prospects. Factors like 'pursue_higher_studies' and 'Skills' also play a crucial role in shaping career paths and income levels. There is a noticeable correlation between study time and the likelihood of class failures (backlogs). The distribution of Base Pay varies across different job roles. Some roles consistently have higher earning potential. Gender disparities might exist in terms of pursuing higher studies, study time, and tuition fees. Further investigation is needed to understand these disparities. The model predicts job roles based on student demographics with a reasonable accuracy. The model predicts graduation marks based on factors like backlogs, pursue_higher_studies, absences, SSC, and HSC with an R-squared value of 0.74.")

st.header("RECOMMENDATIONS")
st.write("Provide students with personalized career guidance based on their academic performance, interests, and skills. Encourage students to focus on their studies and minimize absences to improve their overall performance. Highlight the importance of pursuing higher studies for better career opportunities. Offer resources and support for students facing challenges like backlogs. Provide information about different job roles and their associated earning potential. Address potential gender disparities in education and career choices.")
