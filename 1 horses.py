from flask import Flask, render_template, request
import pickle
import pandas as pd


with open("1LGBM.pkl","rb")as file:
    model=pickle.load(file)
scoor={}
with open("process.pkl","rb")as file1:
    transform=pickle.load(file1)
app = Flask(__name__)

@app.route("/pred", methods=["GET","POST"])
def h_prediction():
    if request.method == "POST":
        Distance = 10
        Direction = "Left"
        Surface = "Dirt"
        Condition = "Good"
        Entrants = 5
        Weight_carried = 4
        FF_Overall = 70
        horse_name = request.form.getlist("item")
        print(horse_name)
        scoor = {}  
        for i in range(len(horse_name)): 
            user_input1 = {
                "Distance": Distance,
                "Direction": Direction,
                "Surface": Surface,
                "Condition": Condition,
                "Weight Carried": Weight_carried,
                "FF (Overall)": FF_Overall,
                "Entrants": Entrants,
                "Horses_name": horse_name[i]
            }
            user_input = pd.DataFrame([user_input1])
            user_x = transform.transform(user_input)
            prediction = model.predict(user_x)
            scoor[horse_name[i]] = prediction[0]  
        if scoor:
            winner = min(scoor, key=scoor.get)
            time = scoor[winner]
            return f"Winner Should be {winner}, it will take approximately {time} time to finish the race"
        else:
            return "No horses selected."

    return render_template("horses_pred.html")

if __name__ == "__main__":
    app.run(debug=False)
    
    