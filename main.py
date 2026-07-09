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
from ml_engine import KeplerMLEngine

# Initialize shared ML engine
engine = KeplerMLEngine()

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
        
import matplotlib.pyplot as plt

def EDA_Dataset():
    global dataset
    global X, y
    
    # --- Correlation Heatmap ---
    num_df = dataset.select_dtypes(include=['number'])
    corr = num_df.corr()
    
    plt.matshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Heatmap', pad=20)
    plt.show()
    
    # --- Countplot for koi_score ---
    if 'koi_score' in dataset.columns:
        counts = dataset['koi_score'].value_counts()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel('koi_score')
        plt.ylabel('Count')
        plt.title('Distribution of koi_score')
        plt.show()
    else:
        print("Column 'koi_score' not found in dataset.")
       
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
    
    if os.path.exists('models/KNN_regression.pkl'):
        regression = joblib.load('models/KNN_regression.pkl')
        text.insert(END,"Loaded saved KNN regression model.")
    else:
        regression = KNeighborsRegressor(n_neighbors=20)
        regression.fit(x_train, y_train)
        joblib.dump(regression, 'models/KNN_regression.pkl')
        text.insert(END,"KNN regression model trained and saved.")
    
    predict = regression.predict(x_test)
    calculateRegressionMetrics("KNN Regression", predict, y_test)

def Linear_Regression():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)  

    modelfile = 'models/LR.pkl'  
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
    model = 'models/catboost_regressor.pkl'
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
            
    model_path='models/catboost_regressor.pkl'
    cat_regressor=joblib.load(model_path)
    pred_Score= cat_regressor.predict(TestData)
    TestData["Pred_Score"]=pred_Score
    text.insert(END, f'{TestData}')
    


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



from tkinter import *
from PIL import Image, ImageTk

# =========================================================
# WINDOW SETUP
# =========================================================
main = Tk()
main.title("Exoplanet Validation ML System")

screen_w = main.winfo_screenwidth()
screen_h = main.winfo_screenheight()
main.geometry(f"{screen_w}x{screen_h}")

# =========================================================
# BACKGROUND IMAGE
# =========================================================
bg_img = Image.open("kepler.png")
bg_img = bg_img.resize((screen_w, screen_h), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# =========================================================
# TITLE RIBBON
# =========================================================
title = Label(
    main,
    text="Machine Learning Based Modelling of Exoplanet Validation Scores from Kepler Telescope Data",
    font=("Segoe UI", 16, "bold"),
    bg="#D7BDE2",
    fg="black",
    pady=10
)
title.place(relx=0.5, rely=0.02, anchor="n", relwidth=0.92)

# =========================================================
# UI GRID CONSTANTS
# =========================================================
BTN_W = 0.15
BTN_H = 0.055
START_X = 0.05
ROW_GAP = 0.065
START_Y = 0.15

# =========================================================
# BUTTON STYLE ENGINE
# =========================================================
def hover_on(e):
    e.widget["bg"] = "#BB8FCE"

def hover_off(e):
    e.widget["bg"] = "#EBDEF0"

def styled_button(text, cmd, y, color="#EBDEF0"):
    btn = Button(
        main,
        text=text,
        command=cmd,
        font=("Segoe UI", 11, "bold"),
        bg=color,
        fg="black",
        activebackground="#1F618D",
        activeforeground="white",
        relief="flat",
        cursor="hand2"
    )
    btn.place(relx=START_X, rely=y, relwidth=BTN_W, relheight=BTN_H)

    btn.bind("<Enter>", hover_on)
    btn.bind("<Leave>", hover_off)

# =========================================================
# BUTTON LAYOUT
# =========================================================
styled_button("Dataset", uploadDataset, START_Y + ROW_GAP*0)
styled_button("Preprocessing", Preprocess_Dataset, START_Y + ROW_GAP*1)
styled_button("EDA", EDA_Dataset, START_Y + ROW_GAP*2)
styled_button("Train/Test Split", Train_Test_Splitting, START_Y + ROW_GAP*3)
styled_button("KNN Regression", knn_regressor, START_Y + ROW_GAP*4)
styled_button("Linear Regression", Linear_Regression, START_Y + ROW_GAP*5)
styled_button("Huber Regression", Huber_regressor, START_Y + ROW_GAP*6)
styled_button("CatBoost Regression", Cat_Boost_regressor, START_Y + ROW_GAP*7)
styled_button("Prediction", Prediction, START_Y + ROW_GAP*8, "#ABEBC6")
styled_button("Comparison Graph", graph, START_Y + ROW_GAP*9)
styled_button("Exit", main.destroy, START_Y + ROW_GAP*10, "#EC7063")

# =========================================================
# WORKSPACE PANEL
# =========================================================

# ================= TEXT WORKSPACE PANEL =================
workspace_frame = Frame(main, bg="white", bd=3, relief="ridge")
workspace_frame.place(
    relx=0.26,
    rely=0.25,
    relwidth=0.48,
    relheight=0.48
)

text_font = ('times', 12, 'bold')

text = Text(
    workspace_frame,
    font=text_font,
    wrap="word",
    bg="white",
    fg="black"
)

scroll = Scrollbar(workspace_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

scroll.pack(side=RIGHT, fill=Y)
text.pack(fill=BOTH, expand=True)


# =========================================================
# RUN APP
# =========================================================
main.mainloop()
