import streamlit as st
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga



st.set_page_config(page_title="Aplicação para otimização de transporte de carga", layout="wide")
st.title("Otimização de transporte de carga")

def load_data(file):
    return pd.read_csv(file, sep=";")

def fitness_function(X, data, max_weigth, max_volume):
    selected_items = data.iloc[X.astype(bool),:]
    total_weigth = selected_items['PESO'].sum()
    total_volume = selected_items['VOLUME'].sum()
    if total_weigth > max_weigth or total_volume > max_volume:
        return -1
    else:
        return - selected_items['VALOR'].sum()
    
data=None


col1, col2 = st.columns(2)

with col1.expander("Carregamento dos dados"):
    uploaded_file = st.file_uploader("Escolha o arquivo", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        carregar = st.button("Carregar os Dados")
        if carregar:
            st.write(data)
            st.write(f"Total de Itens: {len(data)}")
            st.write(f"Peso Total: {data['PESO'].sum()}")
            st.write(f"Volume Total: {data['VOLUME'].sum()}")
            st.write(f"Valor Total: {data['VALOR'].sum()}")


with col2.expander("Processamento"):
    with st.spinner("Aguarde um instante.... processando"):
        peso_maximo = st.number_input("Peso Máximo", value=5000)
        volume_maximo = st.number_input("Volume Máximo", value=300)
        iteracoes = st.number_input("Número de Iterações  \'Algoritmo\'", value=20)
        populacao = st.number_input("População  \'Algoritmo\'", value=500)
        processar = st.button("Processar")
        if data is not None and processar:
            algo_params = {
                "max_num_iteration": iteracoes,
                "population_size": populacao,
                "elit_ratio": 0.02,
                "mutation_probability": 0.5,
                "crossover_probability": 0.2,
                "crossover_type": "uniform",
                "parents_portion": 0.3,
                "max_iteration_without_improv": None
            }
            varbound = [[1,2]] * len(data)
            modelo = ga(function=lambda X: fitness_function(X, data, peso_maximo, volume_maximo),
            dimension=len(data),
            variable_type="bool",
            variable_boundaries=varbound,
            algorithm_parameters=algo_params)       
            modelo.run()

            solution = data.iloc[modelo.output_dict['variable'].astype(bool),:]
            st.write(solution)
            st.write(f"Total de Itens a serem transportados: {len(solution)}")
            st.write(f"Peso Total: {solution['PESO'].sum()}")
            st.write(f"Volume Total: {solution['VOLUME'].sum()}")
            st.write(f"Valor Total: {solution['VALOR'].sum()}")








