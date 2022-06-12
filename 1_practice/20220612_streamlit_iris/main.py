# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from pandas.plotting import radviz
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

def set_iris_data():
    df = px.data.iris()
    return df

def show_data_selection_bar(df):
    st.sidebar.title('Selector')
    species = df['species'].unique()
    selected_species = st.sidebar.multiselect(
        'Species', options=species, default=species)
    min_value = df['sepal_length'].min().item()
    max_value = df['sepal_length'].max().item()
    sepal_len_min, sepal_len_max = st.sidebar.slider(
        'Sepal Length', 
        min_value=min_value, max_value=max_value,
        value=(min_value, max_value)
    )
    options = {}
    options['selected_species'] = selected_species
    options['sepal_len_min'] = sepal_len_min
    options['sepal_len_max'] = sepal_len_max
    return options
   
    
def show_dataframe(df):
    st.subheader('Selected Data:')
    st.dataframe(df)
    st.subheader('Data Description:')
    species = df['species'].unique()
    if 'setosa' in species:
        st.caption('Setosa:')
        st.dataframe(df[df['species']=='setosa'].describe())
    if 'versicolor' in species:
        st.caption('Versicolor:')
        st.dataframe(df[df['species']=='versicolor'].describe())
    if 'virginica' in species:
        st.caption('Virginica:')
        st.dataframe(df[df['species']=='virginica'].describe())

        
def show_scattermatrix(df):
    st.subheader('Scatter Matrix:')
    fig = px.scatter_matrix(data_frame=df,color='species')
    st.plotly_chart(fig, use_container_width=True)

def show_scatterplot(df):
    st.subheader('Scatter Plot:')
    # extract column names
    axis_list = df.columns.unique()
    # select X axis name
    selected_xaxis = st.selectbox(
        'X-axis', axis_list, 
    )
    # select Y axis name
    selected_yaxis = st.selectbox(
        'Y-axis', axis_list
    )
    # 
    fig = px.scatter(df, x=selected_xaxis, y=selected_yaxis, color="species")
    st.plotly_chart(fig, use_container_width=True)

def show_boxplot(df):
    st.subheader('Box Plot:')
    st.caption('Sepal_length:')
    fig = px.box(df, x="species", y="sepal_length", points="all", color="species")
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Sepal_width:')
    fig = px.box(df, x="species", y="sepal_width", points="all", color="species")
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Petal_length:')
    fig = px.box(df, x="species", y="petal_length", points="all", color="species")
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Petal_width:')
    fig = px.box(df, x="species", y="petal_width", points="all", color="species")
    st.plotly_chart(fig, use_container_width=True)

def otherplot(df):
    st.subheader('Andrews Curves:')
    fig = andrews_curves(df, "species")
    st.pyplot(fig.figure, use_container_width=True)
    plt.close()
    st.subheader('Parallel coordinates:')
    fig = parallel_coordinates(df, "species")
    st.pyplot(fig.figure, use_container_width=True)
    plt.close()
    st.subheader('Radviz:')
    fig = radviz(df, "species")
    st.pyplot(fig.figure, use_container_width=True)   
    plt.close()


def correlation_matrix(df):
    st.subheader('Correlation:')
    species = df['species'].unique()
    if 'setosa' in species:
        st.caption('Setosa:')
        df_corr = df[df['species']=='setosa'].corr()
        fig = sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True)
        st.pyplot(fig.figure, use_container_width=True)
        plt.close()
    if 'versicolor' in species:
        st.caption('Versicolor:')
        df_corr = df[df['species']=='versicolor'].corr()
        fig = sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True)
        st.pyplot(fig.figure, use_container_width=True)
        plt.close()
    if 'virginica' in species:
        st.caption('Virginica:')
        df_corr = df[df['species']=='virginica'].corr()
        fig = sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True)
        st.pyplot(fig.figure, use_container_width=True)
        plt.close()

def statistical_test(df, test_type='para'):
    if test_type == 'para':
        # parametric
        prehoc = stats.f_oneway(
            df[df['species'] == 'setosa'].drop(['species'], axis=1),
            df[df['species'] == 'versicolor'].drop(['species'], axis=1),
            df[df['species'] == 'virginica'].drop(['species'], axis=1))  # ANOVA test
            
        posthoc = sp.posthoc_tukey(
            df,
            val_col=df.drop(['species'], axis=1).columns[0],
            group_col='species')  # Turkey test
    else:
        # nonpara metric
        prehoc = stats.kruskal(
            df[df['species'] == 'setosa'].drop(['species'], axis=1),
            df[df['species'] == 'versicolor'].drop(['species'], axis=1),
            df[df['species'] == 'virginica'].drop(['species'], axis=1))  # Kruskal-Wallis test
        posthoc = sp.posthoc_dscf(
            df,
            val_col=df.drop(['species'], axis=1).columns[0],
            group_col='species')  # Dwass, Steel, Critchlow and Fligner all-pairs comparison test
    return prehoc, posthoc

def summary_stats(df):
    if len(df['species'].unique()) == 3:
        test_type = 'para'  # para or nonpara
        prehoc_p = []
        posthoc_p01 = []
        posthoc_p02 = []
        posthoc_p12 = []
        # Test
        for col in df.drop(['species', 'species_id'], axis=1).columns:
            data = df.loc[:, [col, 'species']]
            prehoc, posthoc = statistical_test(data, test_type)
            prehoc_p.append(prehoc.pvalue)
            posthoc_p01.append(posthoc['setosa']['versicolor'])
            posthoc_p02.append(posthoc['setosa']['virginica'])
            posthoc_p12.append(posthoc['versicolor']['virginica'])
        result_df = pd.DataFrame(
            {"Variable": df.drop(['species', 'species_id'], axis=1).columns, "Setosa vs Versicolor vs Virginica": prehoc_p, "Setosa vs Versicolor": posthoc_p01, "Setosa vs Virginica": posthoc_p02, "Versicolor vs Virginica": posthoc_p12})
        st.subheader('Statistical test (P-value):')
        if test_type == 'para':
            st.write("Parametric Test: ANOVA & Tukey-Kramer Test")
        else:
            st.write("Non-Parametric Test: Kruskal-Wallis & Steel-Dwass Test")
        st.dataframe(result_df)

def main():
    st.title('Iris Dataset Dashboard')
    df = set_iris_data()
    
    options = show_data_selection_bar(df)
    df_tmp = df[df['species'].isin(options['selected_species'])]
    df_selected = df_tmp[(df_tmp['sepal_length'] >= options['sepal_len_min']) & 
                         (df_tmp['sepal_length'] <= options['sepal_len_max'])]
    
    show_dataframe(df_selected)
    try:
        show_scattermatrix(df_selected)
    except:
        pass
    try:
        show_scatterplot(df_selected)
    except:
        pass
    try:
        show_boxplot(df_selected)
    except:
        pass
    try:
        correlation_matrix(df_selected.drop('species_id', axis=1))
    except:
        pass
    try:
        otherplot(df_selected)
    except:
        pass
    try:
        summary_stats(df_selected)
    except:
        pass

if __name__ == '__main__':
    main()

