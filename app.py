# Import Library
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import dask
import dask.dataframe as dd
import plotly
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff 
import plotly.express as px
import plotly.graph_objects as go 
import json, csv, os, pickle
import math
from math import *


import uuid
secret_key = uuid.uuid4().hex
print(secret_key)


app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = '8da96fb9a89d456aaf47f825eda11429'

# Tentukan folder untuk menyimpan file yang diunggah untuk diproses lebih lanjut
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')

# Konfigurasikan upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Load model 
model_filename = 'model\model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)


# Load preprocessing
prepro_filename = 'model\minmax_scaler.pkl'
with open(prepro_filename, 'rb') as file:
    loaded_prepro = pickle.load(file)


# 1. Load data
fraud = pd.read_csv('dataset/fraudTrain.csv')
fraud = fraud[0:10000]
# 2. menghapus kolom yang tidak relevan
fraud = fraud.drop('Unnamed: 0', axis=1)
# 3. Mengubah kolom 'dob' dan 'trans_date_trans_time' menjadi datetime
fraud.dob = pd.to_datetime(fraud['dob'])
fraud.trans_date_trans_time = pd.to_datetime(fraud['trans_date_trans_time'])
# 4. Membuat sebuah kolom baru "transaction date" lalu kolom menjadi datetime
fraud['transaction_date'] = pd.to_datetime(fraud['trans_date_trans_time'], format='%Y:%M:%D').dt.date
fraud.transaction_date = pd.to_datetime(fraud['transaction_date'])
# 5. Membuat sebuah kolom "age"
fraud["age"] = pd.DatetimeIndex(fraud["trans_date_trans_time"]).year-pd.DatetimeIndex(fraud["dob"]).year
# 6. Membuat sebuah kolom "year"
fraud["year"] = pd.DatetimeIndex(fraud["trans_date_trans_time"]).year.astype(int).astype(str)
# 7.a. Membuat sebuah kolom "day of the week"
fraud["day_of_week"] = pd.DatetimeIndex(fraud["trans_date_trans_time"]).dayofweek + 1
# 7.b. Membuat sebuah kolom "day"
fraud["day"] = pd.DatetimeIndex(fraud['trans_date_trans_time']).day
# 7.c. Membuat sebuah kolom "hour"
fraud["hour"] = pd.DatetimeIndex(fraud["trans_date_trans_time"]).hour
# 7.d. Membuat sebuah kolom "month"
fraud["month"] = pd.DatetimeIndex(fraud.trans_date_trans_time).month
print(fraud)

# Route untuk halaman awal
@app.route('/')
# Function untuk merender halaman index.html
def home():
    return render_template('index.html')

# Route untuk halaman about
@app.route('/about')
# Function untuk merender halaman about-us.html
def about():
    return render_template('about-us.html')

# Route untuk halaman dashboard -> Predictive
@app.route('/predictive')
# Function untuk merender halaman predictive.html
def predictive():
    return render_template('predictive.html')

# Route untuk halaman dashboard -> Analytics
@app.route('/dashboard')
# Function untuk merender halaman dasshboard.html
def dashboard():
    # Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
    global fig1,fig2, fig3, fig4, fig5, fig6, fig, fig7
    ###===================== Amount Vs Fraud ==========================###
    # Buat plotly histogram
    fig1 = px.histogram(fraud, x='amt', color='is_fraud',
                        nbins=25, opacity=0.7, histnorm='percent',
                        labels={'amt': 'Amount'},
                        color_discrete_sequence=['#1f77b4', '#DE3163'],
                        category_orders={'is_fraud': [0, 1]})

    # Atur layout dan tampilan
    fig1.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=500,
                       height=400, 
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       font=dict(size=14))
    fig1.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig1.update_xaxes(title_text='Amount')
    fig1.update_yaxes(title_text='Percentage')
    fig1.update_xaxes(color='black', showgrid=False)
    fig1.update_yaxes(color='black', showgrid=False)
    fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Age Distribution in Fraudulent vs Non-Fraudulent Transactions ==========================###
    # Pisahkan data antara Fraud dan Non-Fraud
    data_non_fraud = fraud[fraud['is_fraud'] == 0]['age']
    data_fraud = fraud[fraud['is_fraud'] == 1]['age']

    # Buat distplot menggunakan plotly.figure_factory
    fig2 = ff.create_distplot([data_non_fraud, data_fraud], group_labels=['Not Fraud', 'Fraud'],
                            bin_size=2, colors=['#1f77b4', '#DE3163'])

    # Atur layout dan tampilan
    fig2.update_layout(xaxis_title='Age',
                       yaxis_title='Density')
    fig2.update_layout(margin=dict(l=100, r=0, t=0, b=0), 
                       width=460,
                       height=400, 
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       font=dict(size=14))
    fig2.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig2.update_traces(hoverlabel_font_color='white')
    fig2.update_xaxes(color='black', showgrid=False)
    fig2.update_yaxes(color='black', showgrid=False)
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Day of Week Vs Fraud ==========================###
    # Buat plotly histogram
    fig3 = px.histogram(fraud, x='day_of_week', color='is_fraud',
                        nbins=7, opacity=0.7, histnorm='percent',
                        color_discrete_sequence=['#1f77b4', '#DE3163'],
                        category_orders={'is_fraud': [0, 1]},
                        )

    # Atur layout dan tampilan
    fig3.update_layout(margin=dict(l=100, r=0, t=0, b=0), 
                       width=420,
                       height=400, 
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       font=dict(size=14))
    fig3.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig3.update_xaxes(title_text='Day of Week')
    fig3.update_yaxes(title_text='Percentage')
    fig3.update_xaxes(color='black', showgrid=False)
    fig3.update_yaxes(color='black', showgrid=False)
    fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    
    ###===================== Hour Vs Fraud ==========================###
    # Buat plotly histogram
    fig4 = px.histogram(fraud, x='hour', color='is_fraud',
                        nbins=7, opacity=0.7, histnorm='percent',
                        color_discrete_sequence=['#1f77b4', '#DE3163'],
                        category_orders={'is_fraud': [0, 1]},
                        )

    # Atur layout dan tampilan
    fig4.update_layout(margin=dict(l=100, r=0, t=0, b=0), 
                       width=420,
                       height=400, 
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       font=dict(size=14))
    fig4.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig4.update_xaxes(title_text='Hour')
    fig4.update_yaxes(title_text='Percentage')
    fig4.update_xaxes(color='black', showgrid=False)
    fig4.update_yaxes(color='black', showgrid=False)
    fig4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Fraudulent vs Non-Fraudulent Transactions in each Categories ==========================###
    # Hitung jumlah untuk setiap kombinasi 'is_fraud' dan 'kategori'
    counts = fraud.groupby(['is_fraud', 'category']).size().reset_index(name='Count')

    # Membuat diagram batang dua arah
    fig5 = go.Figure()

    # Nilai Positif (Fraud = 1)
    positive_data = counts[counts['is_fraud'] == 1]
    fig5.add_trace(go.Bar(
        x=positive_data['Count'],
        y=positive_data['category'],
        orientation='h',
        name='Fraud',
        marker=dict(color='#DE3163')
    ))

    # Nilai Negatif (Not Fraud = 0)
    negative_data = counts[counts['is_fraud'] == 0]
    fig5.add_trace(go.Bar(
        x=-negative_data['Count'],  
        y=negative_data['category'],
        orientation='h',
        name='Not Fraud',
        marker=dict(color='#1f77b4')
    ))
    # Atur layout dan tampilan
    fig5.update_layout(margin=dict(l=100, r=0, t=0, b=0), 
                       width=500,
                       height=400, 
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       font=dict(size=14))
    fig5.update_layout(
        barmode='relative',  
        xaxis_title='Count',
        yaxis_title='Categories',
    )
    fig5.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig5.update_xaxes(color='black', showgrid=False)
    fig5.update_yaxes(color='black', showgrid=False)
    fig5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    ###===================== Fraudulent vs Non-Fraudulent Transactions Distribution ==========================###
    
    count_of_classes = fraud['is_fraud'].value_counts().sort_index()
    total = float(len(fraud))

    # Hitung persentase
    percentage_of_classes = count_of_classes / total * 100

    # Plot bar chart
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=['Not Fraud', 'Fraud'],
        y=count_of_classes.values,
        text=[f'{y} ({p:.2f}%)' for x, y, p in zip(count_of_classes.index, count_of_classes.values, percentage_of_classes)],
        textposition='outside',
        hoverinfo='text',
        marker=dict(color=['#1f77b4', '#DE3163'])
    ))

    # Atur layout dan tampilan
    fig6.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                       width=460,
                       height=400,
                       margin=dict(l=100, r=0, t=0, b=0),
                       font=dict(size=14))
    fig6.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig6.update_traces(hoverlabel_font_color='white')
    fig6.update_xaxes(color='black', showgrid=False)
    fig6.update_yaxes(color='black', showgrid=False)
    fig6.update_xaxes(title_text='Fraud')
    fig6.update_yaxes(title_text='Count')
    fig6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    
    ###===================== Gender Vs Fraud ==========================###
    # Buat plotly bar chart
    columns = ['gender']
    width_per_subplot = 360
    height_per_subplot = 400
    total_width = 2 * width_per_subplot
    total_height = len(columns) * height_per_subplot

    fig = sp.make_subplots(rows=len(columns), cols=2, subplot_titles=['Fraud = 0 (Not Fraud)', 'Fraud = 1 (Fraud)'],
                        shared_yaxes=True, vertical_spacing=0.1,
                        column_width=[0.5, 0.5])
    
    for i, col in enumerate(columns, start=1):
        value_counts_churn = fraud.groupby([col, 'is_fraud']).size().unstack().fillna(0).stack().reset_index(name='Count')
        unique_values = value_counts_churn[col].unique()
        for value in unique_values:
            trace_0 = go.Bar(x=[value],
                            y=[value_counts_churn[(value_counts_churn[col] == value) & (value_counts_churn['is_fraud'] == 0)]['Count'].iloc[0]],
                            name=f'{value} - Fraud = 0 (Not Fraud)',
                            marker=dict(color='#1f77b4'))
            trace_1 = go.Bar(x=[value],
                            y=[value_counts_churn[(value_counts_churn[col] == value) & (value_counts_churn['is_fraud'] == 1)]['Count'].iloc[0]],
                            name=f'{value} - Fraud = 1 (Fraud)',
                            marker=dict(color='#DE3163'))
            fig.add_trace(trace_0, row=i, col=1)
            fig.add_trace(trace_1, row=i, col=2)

        fig.update_xaxes(title_text=f'{col}', row=i, col=1)
        fig.update_xaxes(title_text=f'{col}', row=i, col=2)
        fig.update_traces(hovertemplate=f'{col}: %{{x}}<br>Jumlah Data: %{{y}}', row=i, col=1)
        fig.update_traces(hovertemplate=f'{col}: %{{x}}<br>Jumlah Data: %{{y}}', row=i, col=2)

        for value in unique_values:
            total_count_0 = value_counts_churn[(value_counts_churn[col] == value) & (value_counts_churn['is_fraud'] == 0)]['Count'].iloc[0]
            total_count_1 = value_counts_churn[(value_counts_churn[col] == value) & (value_counts_churn['is_fraud'] == 1)]['Count'].iloc[0]

            fig.add_annotation(
                x=value, y=total_count_0,  
                text=f'{total_count_0}',
                showarrow=False,
                font=dict(color='black', size=12),  
                xref=f'paper', yref=f'paper', 
                row=i, col=1
            )

            fig.add_annotation(
                x=value, y=total_count_1,  
                text=f'{total_count_1}',
                showarrow=False,
                font=dict(color='black', size=12),  
                xref=f'paper', yref=f'paper', 
                row=i, col=2
            )

    # Atur layout dan tampilan
    fig.update_layout(title_font=dict(color='black'), showlegend=False,
                      width=total_width, height=total_height)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig.update_traces(hoverlabel_font_color='black')
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)
    fig.update_yaxes(title_text='Jumlah Data', row=len(columns)//2, col=1)
    fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Menggunakan scatter_geo untuk peta berbasis latitude dan longitude
    fig7 = px.scatter_geo(fraud, 
                          lat='lat', 
                          lon='long', 
                          color='amt',
                          hover_name='category',
                          color_continuous_scale=px.colors.sequential.Plasma)

    # Atur layout dan tampilan
    fig7.update_layout(margin=dict(l=50, r=0, t=0, b=0), 
                    width=1000,
                    height=400, 
                    font=dict(size=14) 
                    )
    fig7.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig7.update_xaxes(color='black', showgrid=False)
    fig7.update_yaxes(color='black', showgrid=False)
    fig7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)

    # Return render template yang mana akan merender halaman testing
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('dashboard.html', 
                           fig1=fig1,
                           fig2=fig2,
                           fig3=fig3,
                           fig4=fig4,
                           fig5=fig5,
                           fig6=fig6,
                           fig7=fig7,
                           fig=fig)
                           

# Membuat sebuah fungsi untuk menghitung jarak antara lokasi customer dan lokasi merchant
def haversineDistance(lat1,lon1,lat2,lon2):
    Lat_Dist = radians(lat2 - lat1)
    Long_Dist = radians(lon2 - lon1)

    ans = (pow(math.sin(Lat_Dist / 2), 2) + pow(math.sin(Long_Dist / 2), 2) * math.cos(lat1) * math.cos(lat2));
    radius = 6371
    cal = 2 * math.asin(math.sqrt(ans))
    return radius * cal


# Route untuk halaman prediction file yang menjalankan method GET dan POST
# POST mengirim data yang diupload user ke bagian sistem (Backend) untuk diolah
# Data yang telah diolah oleh sistem yang kemudian dikirimkan kembali ke sisi client (Frontend) untuk ditampilkan
@app.route('/prediction', methods=['GET','POST'])
def prediction_file():
    # Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
    global fig1, fig2, fig3, new_results_dict
    # Periksa apakah file sudah diunggah
    if request.method == 'POST':
        # Dapatkan file yang diunggah
        uploaded_df = request.files['file']
        # Dapatkan nama file dan simpan file ke UPLOAD_FOLDER
        data_filename = secure_filename(uploaded_df.filename)
        # Simpan jalur file yang diunggah di sesi
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        filepath = session['uploaded_data_file_path']

        # Muat data dari file CSV ke dalam variabel dict_data dengan format list
        dict_data = []
        with open(filepath, encoding="latin-1") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                dict_data.append(row)

        # Ubah daftar kamus menjadi Pandas DataFrame
        df_results = pd.DataFrame.from_dict(dict_data)
        df_results = df_results[0:20000]
        print(df_results)
        # Mengonversi kolom stempel waktu menjadi datetime
        df_results['dob'] = pd.to_datetime(df_results['dob'])
        df_results['trans_date_trans_time'] = pd.to_datetime(df_results['trans_date_trans_time'])
        df_results['transaction_date'] = pd.to_datetime(df_results['trans_date_trans_time']).dt.date

        # Membuat kolom tambahan
        df_results['age'] = (df_results['trans_date_trans_time'].dt.year - df_results['dob'].dt.year).astype(int)
        df_results['year'] = df_results['trans_date_trans_time'].dt.year.astype(int).astype(str)
        df_results['day_of_week'] = df_results['trans_date_trans_time'].dt.dayofweek + 1
        df_results['day'] = df_results['trans_date_trans_time'].dt.day
        df_results['hour'] = df_results['trans_date_trans_time'].dt.hour
        df_results['month'] = df_results['trans_date_trans_time'].dt.month

        # Ubah kolom yang relevan menjadi tipe data numerik
        numeric_cols = ['lat', 'long', 'merch_lat', 'merch_long']
        df_results[numeric_cols] = df_results[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Hitung jarak menggunakan fungsi haversineDistance
        Dist = []
        for a, b, c, d in zip(df_results['lat'], df_results['long'], df_results['merch_lat'], df_results['merch_long']):
            Dist.append(haversineDistance(a, b, c, d))

        df_results['dist'] = Dist

        # Copy ke variabel dataframe baru
        df_modeling = df_results.copy()

        # Mengeluarkan kolom yang tidak diperlukan
        df_modeling.drop(["trans_date_trans_time", "transaction_date",  "cc_num",
                          "merchant","first","last","street","dob","trans_num",
                          "job","unix_time","state","city","zip","lat", "long",
                          "merch_lat","merch_long"], axis=1, inplace=True)
        
        # Mengambil spesifik kolom saja
        df_modeling = df_modeling[['category','amt','gender','city_pop','age',
                                   'year',	'day_of_week', 'day', 'hour', 'month', 
                                   'dist']]
        
        # Membuat kolom gender menjadi biner dalam numerik
        df_modeling['gender'] = df_modeling['gender'].map({'F':1, 'M':0})

        # Membuat kolom category menjadi kategori dalam numerik
        labelencoder = LabelEncoder()
        df_modeling['category'] = labelencoder.fit_transform(df_modeling['category'])
        print(df_modeling)

        # Preprocess data 
        df_scaled = loaded_prepro.transform(df_modeling)

        # Membuat prediksi menggunakan model yang dimuat
        predictions = loaded_model.predict(df_scaled)

        # Menambahkan prediksi ke DataFrame
        df_results['predictedFraud'] = predictions

        # Mengkonversi pandas dataframe menjadi dictionary
        new_results_dict = df_results.to_dict('records')


        ###===================== Fraudulent vs Non-Fraudulent Transactions Distribution Bar Chart ==========================###
        count_of_classes = df_results['predictedFraud'].value_counts().sort_index()
        total = float(len(df_results))
        # Hitung persentase
        percentage_of_classes = count_of_classes / total * 100

        # Plot bar chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=['Not Fraud', 'Fraud'],
            y=count_of_classes.values,
            text=[f'{y}' for x, y, p in zip(count_of_classes.index, count_of_classes.values, percentage_of_classes)],
            textposition='outside',
            hoverinfo='text',
            marker=dict(color=['#1f77b4', '#DE3163'])
        ))

        # Atur layout dan tampilan
        fig1.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                        width=460,
                        height=400,
                        margin=dict(l=100, r=0, t=0, b=0))
        fig1.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig1.update_traces(hoverlabel_font_color='white')
        fig1.update_xaxes(color='black', showgrid=False)
        fig1.update_yaxes(color='black', showgrid=False)
        fig1.update_xaxes(title_text='Fraud')
        fig1.update_yaxes(title_text='Count')
        fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        ###===================== Fraudulent vs Non-Fraudulent Transactions Distribution Pie Chart ==========================###
        count_of_classes = df_results['predictedFraud'].value_counts().sort_index()
        total = float(len(df_results))
        percentage_of_classes = count_of_classes / total * 100

        # Plot pie chart
        fig2 = go.Figure()
        fig2.add_trace(go.Pie(
            labels=['Not Fraud', 'Fraud'],
            values=count_of_classes.values,
            textinfo='label+percent',
            hoverinfo='label+percent',
            marker=dict(colors=['#1f77b4', '#DE3163'])
        ))

        # Atur layout dan tampilan
        fig2.update_layout(legend=dict(orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.5),
                        width=460,
                        height=400,
                        margin=dict(l=75, r=0, t=0, b=0))
        fig2.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig2.update_traces(hoverlabel_font_color='white')

        fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        
        ###===================== Fraudulent vs Non-Fraudulent Transactions Distribution Bubble Chart ==========================###
        fig3 = px.scatter(df_results, x="cc_num", y="amt", color="category", size="predictedFraud",
                          hover_name="category")
        fig3.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGray')),
                        selector=dict(mode='markers'))

        # Atur layout dan tampilan
        fig3.update_layout(legend=dict(orientation="h", yanchor="top", y=1.6, xanchor="center", x=0.5),
                           width=1000,
                           height=400,
                           margin=dict(l=0, r=0, t=0, b=0))
        fig3.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig3.update_traces(hoverlabel_font_color='white')
        fig3.update_xaxes(color='black', showgrid=False)
        fig3.update_yaxes(color='black', showgrid=False)
        fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Return render template yang mana akan merender halaman testing
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('result.html',
                           fig1=fig1,
                           fig2=fig2,
                           fig3=fig3,
                           data=new_results_dict)

if __name__ == "__main__":
    app.run(debug=True)