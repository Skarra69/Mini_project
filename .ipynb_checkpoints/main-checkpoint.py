from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import matplotlib.pyplot as plt
import joblib
from tkinter import *
from PIL import Image, ImageTk
 
global x_train,y_train,x_test,y_test
global metrics_dict
metrics_dict ={}

def uploadDataset(): 
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")
    

def Preprocess_Dataset():
    text.delete('1.0', END)
    global dataset
    global X,y
    text.delete('1.0', END)
    le =  LabelEncoder()
    for col in dataset.columns:
        dataset.select_dtypes(include='object').columns
        dataset[col]=le.fit_transform(dataset[col])
        X = dataset.drop('koi_score', axis = 1)
        y = dataset['koi_score']
        text.insert(END,str(X.head())+"\n\n")
        text.insert(END,str(y)+"\n\n")
        
def EDA_Dataset():
    global dataset
    global X,y
    text.delete('1.0', END)
    ax = sns.countplot(x = 'koi_pdisposition', data = dataset)
    plt.title("Count Plot")  
    plt.xlabel("koi_pdisposition")  
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center',va='center', fontsize=10, color='black', xytext=(0, 5),textcoords='offset points')   
    plt.show()
       
def Train_Test_Splitting():
    text.delete('1.0', END)
    global X,y
    global x_train,y_train,x_test,y_test

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

    
def calculateRegressionMetrics(algorithm, predict, testY):
    global metrics_dict
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)
    
    # Save metrics for graph
    metrics_dict[algorithm] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    text.insert(END,f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}\n")
    text.insert(END,f"{algorithm} Mean Squared Error (MSE): {mse:.2f}\n")
    text.insert(END,f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}\n")
    text.insert(END,f"{algorithm} R2 Score: {r2:.2f}\n")
    
    plt.figure(figsize=(7, 7))
    plt.scatter(testY, predict, color='blue', alpha=0.6)
    plt.plot([min(testY), max(testY)], [min(testY), max(testY)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")
    plt.title(f"{algorithm} Predicted vs Actual Values")
    plt.grid(True)
    plt.show()

def knn_regressor():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    
    if os.path.exists('KNN_regression.pkl'):
        regression = joblib.load('KNN_regression.pkl')
        text.insert(END,"Loaded saved KNN regression model.")
    else:
        regression = KNeighborsRegressor(n_neighbors=20)
        regression.fit(x_train, y_train)
        joblib.dump(regression, 'KNN_regression.pkl')
        text.insert(END,"KNN regression model trained and saved.")
    
    predict = regression.predict(x_test)
    calculateRegressionMetrics("KNN Regression", predict, y_test)

def Linear_Regression():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)  

    modelfile = 'LR.pkl'  
    if os.path.exists(modelfile):
    
        model = joblib.load(modelfile)
        text.insert(END,"Model loaded successfully.")
        predict = model.predict(x_test)
        calculateRegressionMetrics("Linear Regression", predict, y_test)
    else:
        model = LinearRegression()
        model.fit(x_train, y_train)
        joblib.dump(model, modelfile) 
        text.insert(END,"Model saved successfully.")
        predict = model.predict(x_test)
        calculateRegressionMetrics("Linear Regression", predict, y_test)

    
def Huber_regressor():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    model = 'models/huber_regressor.pkl'
    os.makedirs(os.path.dirname(model), exist_ok=True)

    if os.path.exists(model):
        huber = joblib.load(model)
        text.insert(END,"Loaded saved Huber regression model.")
    else:
        huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100)
        huber.fit(x_train, y_train)
        joblib.dump(huber, model)
        text.insert(END,"Huber regression model trained and saved.")
    predict = huber.predict(x_test)
    calculateRegressionMetrics("Huber Regressor", predict, y_test)

        
def Cat_Boost_regressor():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    model = 'model/catboost_regressor.pkl'
    os.makedirs(os.path.dirname(model), exist_ok=True)

    cat_params = {
        'iterations': 1200,
        'depth': 8,
        'learning_rate': 0.03,
        'loss_function': 'RMSE',
        'colsample_bylevel': 0.8,
        'bootstrap_type': 'Bayesian',
        'random_strength': 2,
        'l2_leaf_reg': 3,
        'bagging_temperature': 1,
        'eval_metric': 'R2',
        'random_seed': 63,
        'verbose': False
}

    if os.path.exists(model):
        cat = joblib.load(model)
        text.insert(END, "Loaded saved CatBoost Regressor model.")

    else:
        cat = CatBoostRegressor(**cat_params)
    
        cat.fit(
            x_train,
            y_train,
            eval_set=(x_test, y_test),
            early_stopping_rounds=50,
            verbose=False
    )

    predict = cat.predict(x_test)
    joblib.dump(cat, model)

    text.insert(END,"CatBoost Regressor trained and saved successfully.")
    calculateRegressionMetrics("CatBoost Regressor", predict, y_test)

def Prediction():
    global TestData, pred_Score

    filename = filedialog.askopenfilename(initialdir="TestData")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')

    TestData=pd.read_csv(filename)

    le =  LabelEncoder()
    for col in TestData.columns:
        if TestData[col].dtypes=='object':
            TestData[col]=le.fit_transform(TestData[col])
            
    model_path='model/catboost_regressor.pkl'
    cat_regressor=joblib.load(model_path)
    pred_Score= cat_regressor.predict(TestData)
    TestData["Pred_Score"]=pred_Score
    text.insert(END, f'{TestData}')
    #for index, row in TestData.iterrows():
        #text.insert(END, f'Row {index + 1}: {row.to_dict()} - Pred_Score: {row["pred_Score"]}\n\n\n\n\n')


def graph():
    global metrics_dict

    text.delete('1.0', END)

    # Check if models are trained
    if not metrics_dict:
        text.insert(END, "Train models first to see comparison graph.\n")
        return

    # Extract R2 scores
    models = []
    r2_scores = []

    for model, metrics in metrics_dict.items():
        models.append(model)
        r2_scores.append(metrics['R2'])

    # Find best model
    best_index = np.argmax(r2_scores)
    best_model = models[best_index]
    best_score = r2_scores[best_index]

    # Plot
    plt.figure(figsize=(8,6))
    bars = plt.bar(models, r2_scores)

    plt.title("Model Comparison Based on R² Score")
    plt.xlabel("Regression Models")
    plt.ylabel("R² Score")
    plt.xticks(rotation=20)

    # Annotate scores on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f'{height:.3f}',
                 ha='center',
                 va='bottom')

    plt.tight_layout()
    plt.show()

    # Show best model in GUI
    text.insert(END, f"Best Model Based on R² Score:\n")
    text.insert(END, f"{best_model} → R² = {best_score:.4f}\n")



def close():
    main.destroy()


main = Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# ---------------- BACKGROUND IMAGE ----------------
def setBackground():
    global bg_photo
    image_path = r"C:\Users\Akhila\Downloads\_0b8b2b70-4c1a-4bf9-9d41-46968bd628ae.jpg" 
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()

font = ('times', 15, 'bold')
title = Label(main, text='Exoplanet Candidate Scoring on Kepler Light Curve and Planetary Features')
title.config(bg='lavender', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Dataset", command=uploadDataset,bg="white",fg="black",activebackground="white",activeforeground="black")
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="Preprocessing", command=Preprocess_Dataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

EDAButton = Button(main, text="EDA", command=EDA_Dataset)
EDAButton.place(x=20,y=200)
EDAButton.config(font=ff)

Button1 = Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
Button1.place(x=20,y=250)
Button1.config(font=ff)

Button2 = Button(main, text="knn Regression", command= knn_regressor)
Button2.place(x=20,y=300)
Button2.config(font=ff)

Button3 = Button(main, text=" Linear Regression", command= Linear_Regression )
Button3.place(x=20,y=350)
Button3.config(font=ff)


Button4 = Button(main, text=" Huber Regression", command=Huber_regressor  )
Button4.place(x=20,y=400)
Button4.config(font=ff)

Button5 = Button(main, text=" CatBoost Regression", command=Cat_Boost_regressor )
Button5.place(x=20,y=450)
Button5.config(font=ff)

Button1 = Button(main, text="Prediction", command=Prediction)
Button1.place(x=20,y=500)
Button1.config(font=ff)

Button1 = Button(main, text="Comparison Graph", command=graph)
Button1.place(x=20,y=550)
Button1.config(font=ff)

Button1 = Button(main, text="Exit", command=close)
Button1.place(x=20,y=600)
Button1.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)


main.config(bg='DarkSlateGray1')
main.mainloop()

