import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

my_blue="#0064B2";
my_red="#D61600";

def accuracy_assessment(param_y_test, param_y_pred):
    accuracy = accuracy_score(param_y_test, param_y_pred)
    print("Accuracy:", accuracy)

    balanced_accuracy = balanced_accuracy_score(param_y_test, param_y_pred)
    print("Balanced accuracy:", balanced_accuracy)

    precision = precision_score(param_y_test, param_y_pred, average="weighted")
    print("Precision:", precision)

    recall = recall_score(param_y_test, param_y_pred, average="weighted")
    print("Sensivity (recall):", recall)

    f1 = f1_score(param_y_test, param_y_pred, average="weighted")
    print("F1-Score:", f1)

def train_test_plot(depth, ma_train, ma_test, x_label, title):
    df_pom=pd.DataFrame({'depth':depth, 'ma_train': ma_train, 'ma_test':ma_test})
    fig = px.line(df_pom, x='depth', y=['ma_train', 'ma_test'], markers=True, line_shape='linear',
                labels={'variable': 'Zbiór danych'},
                color_discrete_map={'ma_train': my_blue, 'ma_test': my_red})

    newnames = {'ma_train': 'Dane treningowe', 'ma_test': "Dane testowe"}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                        )
                  )
    # Wyświetlanie legendy
    fig.update_layout(showlegend=True)
    # Dodajemy etykiety i tytuł
    fig.update_layout(
        xaxis=dict(
            title=x_label,
            tickfont=dict(size=16),
            title_font=dict(size=20),
        ),
        yaxis=dict(
            title='Dokładność',
            tickfont=dict(size=16),
            title_font=dict(size=20)
        )
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    fig.write_image("../images/"+title+ ".png", width=1000, height=600, scale=4, format="png")

    
def significant_variables(model, param_x_train, name):

    indeksy = np.where(model.feature_importances_!=0)[0]
    variables= [param_x_train.columns[i] for i in indeksy]
    importances = model.feature_importances_[indeksy]

    # sortowanie
    importances, variables= zip(*sorted(zip(importances, variables), reverse=False))

    # Tworzenie wykresu słupkowego z niezerowymi wartościami
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.barh(variables[:15], importances[:15])

    plt.xlabel("Ważność zmiennej")
    plt.ylabel("Zmienna")
    # fig.write_image("../images/waznosc_"+name+ ".png", width=1000, height=600, scale=4, format="png")
    plt.show()