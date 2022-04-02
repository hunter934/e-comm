import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="XGBoost Classifier Machine Learning",
    page_icon=(Image.open("han.png")),
    layout="wide"
    )

with st.sidebar:
    # Pilih Dataset
    opsi = st.selectbox('Pilih dataset di bawah ini',
                        ['Pilih dataset ...',
                        'Pakai data E-Commerce Churn',
                        'Coming Soon!'])
    
    if (opsi == 'Pakai data E-Commerce Churn'):
        fe = st.selectbox('Terapkan Normalisasi atau Standarisasi',
                        ['Pilih opsi Normalisasi atau Standarisasi...',
                        'Normalisasi data','Standarisasi data'])

        imba = st.selectbox('Terapkan Oversampling atau Undersampling',
                        ['Pilih opsi Oversampling atau Undersampling...',
                        'Oversampling data','Undersampling data'])

        # Parameter
        learning = st.slider('Learning Rate', 0.0, 1.0, 0.2)
        estimator = st.slider('N-Estimator', 100, 2000, 1000)
        depth = st.slider('Max Depth', 1, 10, 6)
        child_weight = st.slider('Min Child Wright', 0.0, 1.0, 0.75)

# Gambar
img = Image.open("XGBoost_logo.png")
st.image(img, width=200)

# Judul
st.title("XGBoost Classifier Machine Learning")

if (opsi == 'Pakai data E-Commerce Churn'):
    df = pd.read_excel('E Commerce Dataset.xlsx', 1)

    # Drop kolom yang terlalu unik
    for i in df.columns:
        if (df[i].nunique()) == (len(df[i])):
            df.drop(columns=i, inplace=True)

    # Isi baris yang kosong dengan modus atau median
    for col in df:
        if df[col].dtypes == 'O':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Menghapus baris yang terduplikasi
    df = df.drop_duplicates()

    # Memfilter outlier dengan IQR
    for col in df.columns:
        if col != 'Churn' and (df[col].dtypes) != 'O':
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            Lower_fence = df[col].quantile(0.25) - (IQR * 3)
            Upper_fence = df[col].quantile(0.75) + (IQR * 3)

            print(f"Lower Fence {col} : ", Lower_fence)
            print(f"Upper Fence {col} : ", Upper_fence, "\n")
            
            filter = ((df[col] < Lower_fence) | (df[col] > Upper_fence))
            df = df[~filter]

    # Melakukan One Hot Encoding
    for col in df.columns:
        if (df[col].dtypes) == 'O':
            df = df.join(pd.get_dummies(df[col], prefix=col))

    # Menghapus data categorical
    for col in df:
        if df[col].dtypes == 'O':
            df.drop(columns=col,inplace=True)

    # Mengubah format ke float
    for col in df.columns:
        df[col] = df[col].astype('float')

    # Menghitung Null Accuracy
    df0 = df[df['Churn']==0]
    df1 = df[df['Churn']==1]
    
    class_1 = df0['Churn'].value_counts() 
    class_2 = df1['Churn'].value_counts()

    null_accuracy = int(class_1) / (int(class_1) + int(class_2))

    # Train & Test split
    a = df.loc[:, df.columns == 'Churn']
    b = df.loc[:, df.columns != 'Churn']

    target = np.array(a)
    feature = np.array(b)

    X_train, y_train = feature, target

    X_train,X_test, y_train,y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.2,
                                                        random_state=7)

    # Membuat model machine learning
    model = XGBClassifier(learning_rate=learning,
                            n_estimators=estimator,
                            max_depth=depth,
                            min_child_weight=child_weight)
    model_over = XGBClassifier(learning_rate=learning,
                                n_estimators=estimator,
                                max_depth=depth,
                                min_child_weight=child_weight)
    model_under = XGBClassifier(learning_rate=learning,
                                n_estimators=estimator,
                                max_depth=depth,
                                min_child_weight=child_weight)
    
    def jumlah_value(dataset):
        counts = pd.DataFrame(dataset).value_counts()
        st.text("Imbalanced Data Check: ")
        st.text(counts)

    if (fe == 'Normalisasi data'):
        # Normalisasi data
        ms = MinMaxScaler()
        X_train = ms.fit_transform(X_train)
        X_test = ms.transform(X_test)

        if (imba == 'Oversampling data'):
            # Oversampling data
            from imblearn.over_sampling import SMOTE
            oversample = SMOTE()
            X_over, y_over = oversample.fit_resample(X_train, y_train)

            # Training model
            model_over.fit(X_over, y_over)
            
            # Prediksi
            y_pred_over = model_over.predict(X_test)
            predictions_over = [round(value) for value in y_pred_over]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_over, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_over)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_over) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        elif (imba == 'Undersampling data'):
            # Undersampling data
            from imblearn.under_sampling import RandomUnderSampler
            undersample = RandomUnderSampler(sampling_strategy='majority')
            X_under, y_under = undersample.fit_resample(X_train, y_train)

            # Training model
            model_under.fit(X_under, y_under)
            
            # Prediksi
            y_pred_under = model_under.predict(X_test)
            predictions_under = [round(value) for value in y_pred_under]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_under, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_under)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_under) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        else:
            # Training data
            model.fit(X_train, y_train)

            # Prediksi model
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_train)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

    elif (fe == 'Standarisasi data'):
        # Standarisasi  data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if (imba == 'Oversampling data'):
            # Oversampling data
            from imblearn.over_sampling import SMOTE
            oversample = SMOTE()
            X_over, y_over = oversample.fit_resample(X_train, y_train)

            # Training model
            model_over.fit(X_over, y_over)
            
            # Prediksi
            y_pred_over = model_over.predict(X_test)
            predictions_over = [round(value) for value in y_pred_over]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_over, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_over)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_over) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        elif (imba == 'Undersampling data'):
            # Undersampling data
            from imblearn.under_sampling import RandomUnderSampler
            undersample = RandomUnderSampler(sampling_strategy='majority')
            X_under, y_under = undersample.fit_resample(X_train, y_train)

            # Training model
            model_under.fit(X_under, y_under)
            
            # Prediksi
            y_pred_under = model_under.predict(X_test)
            predictions_under = [round(value) for value in y_pred_under]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_under, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_under)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_under) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        else:
            # Training data
            model.fit(X_train, y_train)

            # Prediksi model
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_train)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

    else:
        # Pilih Oversampling atau Undersampling
        if (imba == 'Oversampling data'):
            # Oversampling data
            from imblearn.over_sampling import SMOTE
            oversample = SMOTE()
            X_over, y_over = oversample.fit_resample(X_train, y_train)

            # Training model
            model_over.fit(X_over, y_over)
            
            # Prediksi
            y_pred_over = model_over.predict(X_test)
            predictions_over = [round(value) for value in y_pred_over]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_over, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_over)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_over) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        elif (imba == 'Undersampling data'):
            # Undersampling data
            from imblearn.under_sampling import RandomUnderSampler
            undersample = RandomUnderSampler(sampling_strategy='majority')
            X_under, y_under = undersample.fit_resample(X_train, y_train)

            # Training model
            model_under.fit(X_under, y_under)
            
            # Prediksi
            y_pred_under = model_under.predict(X_test)
            predictions_under = [round(value) for value in y_pred_under]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred_under, bins=10)
            st.pyplot(fig)

            # Hasil prediksi
            jumlah_value(y_under)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred_under) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

        else:
            # Training data
            model.fit(X_train, y_train)

            # Prediksi model
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Plot Histogram
            fig, ax = plt.subplots()
            ax.hist(y_pred, bins=10)
            st.pyplot(fig)
            
            # Hasil prediksi
            jumlah_value(y_train)
            st.text('Akurasi model adalah: %.4f%%' % (accuracy_score(y_test, y_pred) * 100))
            st.text("Null accuracy adalah: %.4f%%" % (null_accuracy * 100))

elif (opsi == 'Coming Soon!'):
    st.text('Dataset lainnya akan segera menyusul.')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
