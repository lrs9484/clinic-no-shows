import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("clinic_appointments.csv")

# X is the appointment information. y is what actually happened.
X = df.drop(columns="no_show")
y = df["no_show"].eq("no-show").astype(int)

# stratify=y keeps the no-show rate similar in both groups.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Total appointments:", len(df))
print("Total no-shows:", y.sum())
print("Training appointments:", len(y_train), "| No-shows:", y_train.sum())
print("Test appointments:", len(y_test), "| No-shows:", y_test.sum())
# Step 2: Fit the decision tree
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=10,
    random_state=42,
)
tree.fit(X_train, y_train)

print("\nDecision tree rules:")
print(export_text(tree, feature_names=list(X_train.columns), show_weights=True))
# Step 3: Evaluate both policies on the test set
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

baseline_predictions = [0] * len(y_test)  # Predict everyone will show
tree_predictions = tree.predict(X_test)

for name, predictions in [
    ("Baseline", baseline_predictions),
    ("v1", tree_predictions),
]:
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name} — test results")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")
    print(f"Recall: {recall_score(y_test, predictions):.1%}")

    if tp + fp == 0:
        print("Precision: undefined (zero predicted no-shows)")
    else:
        print(f"Precision: {precision_score(y_test, predictions):.1%}")

print(f"\nv1 training accuracy: {tree.score(X_train, y_train):.1%}")
print(f"v1 test accuracy: {tree.score(X_test, y_test):.1%}")
# Step 4: Make booking decisions for two upcoming appointments
def booking_decision(appointment):
    row = appointment.to_frame().T
    prediction = tree.predict(row)[0]
    return "overbook" if prediction == 1 else "hold"


upcoming = pd.DataFrame(
    [
        [21, 1, 27, 9.5, 0, 1, 15],  # D
        [3, 0, 61, 2.0, 1, 0, 9],    # E
    ],
    columns=X_train.columns,
    index=["D", "E"],
)

decisions = pd.DataFrame({
    "Leaf": tree.apply(upcoming),
    "Decision": upcoming.apply(booking_decision, axis=1),
})

print("\nUpcoming appointment decisions:")
print(decisions)
# Step 5: Calculate the financial value of each policy
def policy_value(predictions):
    _, fp, fn, _ = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    return -60 * fp - 120 * fn


baseline_value = policy_value(baseline_predictions)
tree_value = policy_value(tree_predictions)

savings = tree_value - baseline_value
savings_per_appointment = savings / len(y_test)
annual_appointments = 10_000  # Assumed annual volume

print("\nFinancial results on the test set:")
print(f"Baseline: {baseline_value:,.2f} USD total, "
      f"{baseline_value / len(y_test):,.2f} USD per appointment")
print(f"v1: {tree_value:,.2f} USD total, "
      f"{tree_value / len(y_test):,.2f} USD per appointment")
print(f"Savings: ${savings:,.2f} total, ${savings_per_appointment:.2f} per appointment")
print(f"Estimated annual savings at {annual_appointments:,} appointments: "
      f"${savings_per_appointment * annual_appointments:,.2f}")
      