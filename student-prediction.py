import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools  # Needed for iteration over matrix cells
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Load the dataset
df = pd.read_csv('student-scores.csv')

# Instantiate a label encoder
label_encoder = LabelEncoder()

# Encoding 'gender' column (if gender is a column)
df['gender'] = label_encoder.fit_transform(df['gender'])
df = pd.get_dummies(df, columns=['gender', 'career_aspiration'])

# Calculate overall score as the average of individual subject scores
df['overall_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                           'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Create a performance category based on overall score
bins = [0, 50, 80, 100]  # Bins for Low, Average, High categories
labels = ['Low', 'Average', 'High']  # Labels for each category
df['performance_category'] = pd.cut(df['overall_score'], bins=bins, labels=labels, right=False)

# Create the target (y) and feature (X) variables for classification
X_class = df.drop(['id', 'first_name', 'last_name', 'email', 'overall_score', 'performance_category'], axis=1)
y_class = df['performance_category']

# Standardize the features
scaler = StandardScaler()
X_scaled_class = scaler.fit_transform(X_class)

# Split the data into training and testing sets (80% train, 20% test)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled_class, y_class, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model on the training data
log_reg.fit(X_train_class, y_train_class)

# Make predictions on the testing data
y_pred_class = log_reg.predict(X_test_class)

# Output classification metrics
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))

# Generate the confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)

# Plot the confusion matrix using matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Capture the mappable object

ax.set_title('Confusion Matrix')
plt.colorbar(cax)  # Use the captured mappable object for colorbar

# Annotate the confusion matrix with counts
classes = ['Low', 'Average', 'High']
tick_marks = range(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Display the counts on the confusion matrix
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

# Set axis labels
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')

# Show the plot
plt.tight_layout()
plt.show()

# Apply 5-fold cross-validation to the Logistic Regression model
cv_scores = cross_val_score(log_reg, X_scaled_class, y_class, cv=5, scoring='accuracy')

# Print the individual fold scores
print("Cross-validation scores for each fold:", cv_scores)

# Print the average cross-validation score
print("Average cross-validation score:", np.mean(cv_scores))

# --- Content-Based Recommendation System Implementation ---

# Create a feature matrix (X) based on the subjects and overall score
X = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score',
        'english_score', 'geography_score', 'overall_score']]

# Standardize the features for similarity comparison
X_scaled = scaler.fit_transform(X)

# Compute cosine similarity between all students
cosine_sim = cosine_similarity(X_scaled)

# Create a function to recommend students based on cosine similarity
def recommend_students(student_id, cosine_sim_matrix, df, top_n=5):
    # Get the index of the student from the DataFrame
    student_idx = df[df['id'] == student_id].index[0]

    # Get similarity scores for the student
    sim_scores = list(enumerate(cosine_sim_matrix[student_idx]))

    # Sort the students based on similarity score (higher similarity is better)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar students
    top_similar_students = sim_scores[1:top_n + 1]

    # Extract the indices and similarity scores
    recommended_indices = [x[0] for x in top_similar_students]
    recommended_scores = [x[1] for x in top_similar_students]

    # Return the student names and similarity scores
    recommended_students = df.iloc[recommended_indices][['first_name', 'last_name', 'performance_category']]
    recommended_students['similarity_score'] = recommended_scores

    return recommended_students

# Example: Recommend top 5 similar students for a student with ID=1
recommended_students = recommend_students(student_id=1, cosine_sim_matrix=cosine_sim, df=df, top_n=5)
print("\nRecommended Students Based on Similarity:\n", recommended_students)
