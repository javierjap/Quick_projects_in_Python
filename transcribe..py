# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:32:38 2023

@author: q32520

En este código en primer lugar se descargan los modelos whisper utilizados para la transcripción y 
posteriormente se utilizan a través de un pipeline para obtener dichas transcripciones.
Es importante para obtener todo el potencial de whisper-large tener actualizadas la librería transformers.
"""
import logging
import os
import time
import glob
import numpy as np
from bde_huggingface import hf_downloader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, WhisperTokenizer, \
    WhisperFeatureExtractor
from datasets import load_from_disk
from utils import *


def save_whisper_model(model_name):
    """
    Función para descargar los modelos.
    
    Parameters
    ----------
    model_name : (type str) Nombre del modelo a descargar
    
    Returns
    -------
    None.

    """
    model_path = hf_downloader.download_model(model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language="spanish", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    feature_extractor.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)


def transcribe_audios(modelo, paths, config):
    """
    Función para transcribir las respuestas a las pregunta p.25a del cuestionario. Dejo la opción de que se pueda seguir usando whisper-base y 
    whisper-medium pero como algo anecdotico. Los modelos que se deben usar son los large y especialmente el v-3. Las maejoras con estos modelos
    son notables. Es por ello que la parte donde modelizo con estos modelos tiene más detalle (me quedo también con el id del hogar en el diccionario
    a devolver).
    
    Parameters
    ----------
    modelo : (type str) Elegir uno de los posibles modelos descargados entre ['openai/whisper-base', 'openai/whisper-medium'] #'openai/whisper-large-v2'
    muestra : Puede ser toda la muestra completa (audio_dataset, entonces es tipo datasets.arrow_dataset.Dataset o audio_dataset[:]  entonces es dict)
            o puede ser un audio en concreto (audio_dataset[i]["audio"]) o un conjunto de audios (audio_dataset[i:j]["audio"]) también puede ser audio_dataset_pre 
            que contiene una lista con los tensor y la sampling rate de los audios del dataset preprocesado

    Returns
    -------
    Devuelve un diccionario con la transcripcion del audio o audios que han sido introducidos como muestra
    
    Fuente
    ------
    https://discuss.huggingface.co/t/how-to-set-language-in-whisper-pipeline-for-audio-transcription/31482/4

    """

    start = time.time()
    model_path = hf_downloader.download_model(modelo)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language="spanish", task="transcribe")

    asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chunk_length_s=30,
            stride_length_s=(4, 2),
            batch_size=1
        )

    path_audio_dataset = paths['path_audio_dataset']
    output_path = paths['output_audios']

    for survey_wave in config['files'].keys():

        ds_path = os.path.join(path_audio_dataset, survey_wave)
        for question in config['files'][survey_wave]:
            logging.info("Transcribing audios for survey-wave {} and question {}".format(survey_wave, question))
            datapaths = glob.glob(os.path.join(ds_path, "{}*".format(question)))
            for datapath in datapaths:
                data = load_from_disk(datapath)

                arrays = [data['audio'][i]['array'] for i in range(len(data))]
                res = asr_pipe(arrays, return_timestamps=True)
                res = dict(zip(data['hh_id'], res))

                if "pre" in datapath:
                    save_path = os.path.join(os.path.join(output_path, survey_wave), question + '_pre')
                else:
                    save_path = os.path.join(os.path.join(output_path, survey_wave), question)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                with open(os.path.join(save_path, "transcription.json"), 'w', encoding='utf-8') as fp:
                    json.dump(res, fp)

    end = time.time()
    total_time = round((end - start) / 60, 2)
    print("\n" + str(total_time) + " minutos")


def compare_audios(paths, config):
    """
    Función que compara las transcripciones de los audios sin preprocesado y con preprocesado

    Parameters
    ----------
    json_path_sin_pre : (str) path que contiene el archivo .json de las transcripciones de los audios sin preprocesado
    json_path_con_pre : (str) path que contiene el archivo .json de las transcripciones de los audios con preprocesdo
    Returns
    -------
    Devuelve un .json en el cual si la transcripción del audio preprocesado es más larga que la del sin preprocesar se queda 
    con la transcripción del audio preprocesado. En el resto de casos me quedo con la transcripción del audio sin preprocesar.

    """

    output_path = paths['output_audios']

    for survey_wave in config['files'].keys():

        for question in config['files'][survey_wave]:
            logging.info("Comparing transcriptions for survey-wave {} and question {}".format(survey_wave, question))
            pre_path = os.path.join(os.path.join(output_path, survey_wave), question + '_pre')
            path = os.path.join(os.path.join(output_path, survey_wave), question)

            pre_path_json = os.path.join(pre_path, "transcription.json")
            path_json = os.path.join(path, "transcription.json")

            with open(path_json) as f:
                pred_timestamp = json.load(f)

            with open(pre_path_json) as f:
                pred_timestamp_pre = json.load(f)
    
            df_sin = pd.DataFrame.from_dict(pred_timestamp, orient='index')
            df_con = pd.DataFrame.from_dict(pred_timestamp_pre, orient='index')
            df_con = df_con.rename(columns={'text': 'text_pre', 'chunks': 'chunks_pre'})

            df_comparacion = df_sin.join(df_con, how='inner')
            df_comparacion['len_text'] = 0
            df_comparacion['len_text_pre'] = 0
            df_comparacion = df_comparacion.reset_index()
            for i in range(len(df_comparacion['text'])):
                df_comparacion.loc[i, 'len_text'] = len(df_comparacion['text'][i].split())
                df_comparacion.loc[i, 'len_text_pre'] = len(df_comparacion['text_pre'][i].split())
            df_comparacion['mejor_pre'] = np.where(df_comparacion['len_text'] + 10 < df_comparacion['len_text_pre'], 1, 0) #Filas en las que la transcripción del texto preprocesado tiene al menos 10 caracteres más que en la transcripción sin preprocesar
            for i in range(len(df_comparacion['text'])):
                if df_comparacion.loc[i, 'mejor_pre'] == 1:
                    df_comparacion.loc[i, 'text'] = df_comparacion.loc[i, 'text_pre']
                    df_comparacion.at[i, 'chunks'] = df_comparacion.at[i, 'chunks_pre']
            df_comparacion = df_comparacion[['index', 'text', 'chunks', 'mejor_pre']]
            df_comparacion.index = df_comparacion.pop('index')
            new_pred_timestamp = pd.DataFrame.to_dict(df_comparacion, orient='index')

            save_path = os.path.join(path, "transcription_final.json")

            with open(save_path, 'w', encoding='utf-8') as fp:
                json.dump(new_pred_timestamp, fp)
