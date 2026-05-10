from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
car = pd.read_csv("cleaned_cars.csv")

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route("/")
def index():
    companies = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, "Select Company")
    return render_template("index.html", companies=companies, car_model=car_model, years=years, fuel_type=fuel_type)


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    year = int(request.form.get("year"))
    car_model = request.form.get("car_model")
    fuel_type = request.form.get("fuel_type")
    kms_driven = int(request.form.get("kms_driven"))
    print(company, year, car_model, kms_driven, fuel_type)


    prediction = model.predict(pd.DataFrame(
        [[company, year, car_model, fuel_type, kms_driven]],
        columns=['company', 'year', 'name', 'fuel_type', 'kms_driven']
    ))
    # print(prediction)
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)