import joblib
import pandas as pd



# 1. Loading the pkl file.

model = joblib.load("product_category_model.pkl")

print("\nModel loaded successfully!")
print("Type 'exit' at any time to stop.\n")

# 2. PREDICTION LOOP

while True:

    # Ask the user for a product title
    title = input("Enter product title: ")

    # Exit condition
    if title.lower().strip() == "exit":
        print("\nExiting...")
        break

    # Remove unnecessary spaces
    title = title.strip()

    # Check for empty input
    if title == "":
        print("Please enter a product title.\n")
        continue


    # 3. CREATE INPUT DATA


    user_input = pd.Series([title])


    # 4. MAKE PREDICTION


    prediction = model.predict(user_input)[0]


    # 5. DISPLAY RESULT


    print(f"Predicted category: {prediction}")





