#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import jsonify

import io
import os
import pandas as pd
import numpy as np
import difflib
import re

from joblib import load
from xgboost import XGBClassifier

from google.cloud import vision
from google.cloud.vision import types

app = Flask(__name__)

data = [
    {
        'label': 'saint_aubain',
        'longitude': '4.8601335189',
        'latitude': '50.4641862752',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/2bff70f6b52452be4a8307dc849319eb',
        'name': 'Cathedrale Saint Aubain',
        'description': "Vue des facades latérales de la Cathédrale Saint-Aubain.",
        'year': 1930
    },
    {
        'label': 'palais_justice',
        'longitude': '4.8607096609',
        'latitude': '50.4641151676',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/a611e730def42dda7abef74d2a8233e9',
        'name': 'Palais du gouverneur',
        'description': "Faisant face à la cathédrale Saint-Aubain, l'ancien palais épiscopal construit à l'initiative de l'évêque Thomas de Strickland de 1728 à 1730, sur les plans du namurois J.T. Maljean, est une imposante demeure de style classique, en briques et pierre bleue. Il présente un pland en U encadrant une cour fermée à rue par un mur percé d'un grand portail, auquel s'adosse une galerie. Dans l'axe du rez-de-chaussée du corps principal de deux niveaux sous une toiture à la Mansard comme ses deux ailes, le porche ouvert, à pans concaves, constitue un ajout réalisé entre 1772 et 1779 par l'évêque Lobkowitz. A droite du palais, un bâtiment ici de trois niveaux, mais sous même hauteur de corniche abrite les locaux de l'ancienne administration diocésaine. L'intérieur s'ouvre par un vaste hall décoré de stucs signés en 1773 par les frères Moretti, comme l'ancienne chapelle devenue salle du Conseil Provincial, que meublent aussi de grandes peintures murales signées par Marinus et datées de 1851-1863",
        'year': 1906
    },
    {
        'label': 'gare',
        'longitude': '4.8634576715',
        'latitude': '50.4683621913',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/7bafb07d523bb8f9eb16a3ed4679eed1',
        'name': 'La gare',
        'description': "Cette carte envoyée à Budapest en Hongrie montre une collection de voitures bien rangées devant la Gare pendant que leurs propriétaires préfèrent encore le train pour couvrir les longues distances. Les autoroutes n'existaient pas encore.",
        'year': 1930
    },
    {
        'label': 'place_ange',
        'longitude': '4.8651619623',
        'latitude': '50.4643995661',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/d0727233278221fff7a1e90d4c524468',
        'name': 'Place de l''ange',
        'description': "La Place de l'ange au lendemain de la deuxième guerre mondiale. Le très bel Hôtel d'Harscamp avec sa tour en forme de clocher a résisté aux bombardements mais ne résistera pas au modernisme des années 70",
        'year': 1948
    },
    {
        'label': 'place_armes',
        'longitude': '4.8672316348',
        'latitude': '50.4629022412',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/dbee818e40870e194b6960b3a82a55c8',
        'name': 'Place d''armes''',
        'description': "La place d'armes en travaux. Le pavement est en cours de rénovation. Avant le placement de la statue de Leopold II et la disparition du  kiosque à musique ? La Rangée de maisons du fond n'avait pas encore disparu pour laisser place à la rue du beffroi et la rangée de gauche n'avait pas encore été remplacée par l'innovation.",
        'year': 1920
    },
    {
        'label': 'rue_bruxelles',
        'longitude': '4.8609207504',
        'latitude': '50.4663051598',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/cfa04db3f9d5657c8259651814e2f117',
        'name': 'Rue de Bruxelles',
        'description': "Collège devenu aujourd'hui l'Unamur.",
        'year': 1905
    },
    {
        'label': 'felicien_rops',
        'longitude': '4.8626652312',
        'latitude': '50.462829396',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/1adc5e218b2acc9bf4814928576ae842',
        'name': 'Rue fumal',
        'description': "Maisons anciennes dans la rue Fumal.",
        'year': 1930
    },
    {
        'label': 'rue_fer',
        'longitude': '4.8650379186',
        'latitude': '50.4645477016',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/020d31722e34d6713c96ba3ad89fcc74',
        'name': "Croisement entre la rue Haute Marcelle et la rue de l'Ange",
        'description': "Croisement entre la rue Haute Marcelle et la rue de l'Ange.",
        'year': 1930
    },
    {
        'label': 'saint_loup',
        'longitude': '4.8637291992',
        'latitude': '50.4636832954',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/99cb38cbbc046f2f6c0144e5a0e3cbd1',
        'name': 'Eglise Saint Loup',
        'description': "L'édifice de style baroque fut construit entre 1621 à 1645 sur les plans de Frère Huyssens. architecte de la Compagnie de Jésus. pour le collège des Jésuites dont il constituait la chapelle. Dédié à saint Ignace. il fut érigé en paroisse à la fin du XVIIIème siècle. prenant alors le nom de Saint-Loup pour remplacer l'église de ce vocable. démolie au Marché-aux-légumes.La façade en calcaire de Meuse et en pierre blanche a été remontée entre 1860 et 1867 sous la directon de l'architecte Boveroulle.L'intérieur est à signaler par la grandeur et la qualité de son décor où sont à remarquer principalement les plafonds entièrement sculptés en pierre de sable de Maastricht. l'emploi du marbre noir de Nmaur en alternance avec du marbre rouge et les confessionnaux des XVIIème et XVIIème siècles.",
        'year': 1905
    },
    {
        'label': 'theatre',
        'longitude': '4.8674607394',
        'latitude': '50.4641003172',
        'old_photo_url': 'https://data.namur.be/api/v2/catalog/datasets/namur-photos-anciennes/files/18a36db67dd9bb27bc8bb52e5e91d909',
        'name': 'Le Théâtre',
        'description': "Le Théâtre de Namur venait d'être débarrassé de sa verrière jugée disgracieuse. Par contre elle protégeait le morceau de façade qu'elle abritait et qui apparaît en bien meilleur état que le reste+G15. A la droite du Théâtre on distingue une chapelle qui devait faire partie de l'institut Notre-Dame et aujourd'hui disparue.C'est en juillet 1809 que Julie Billiart inaugure l'institut Notre-Dame. selon l'accord scellé avec l'industriel Léopold Zoude. Les soeurs étaient aussi connues sous le vocable de ""dames françaises"". Julie Billiard et ses consoeurs françaises d'origine fuyaient  les révolutionnaires. Cet institut existe toujours aujourd'hui. Elles ont aussi fondé des instituts notre-dame à Bastogne et à Philipeville",
        'year': 1932
    },

]

damn = [[0.0, 0.0, 0.8825677633285522, 0.8487496376037598, 0.0, 0.8659344911575317, 0.8710678219795227, 0.0,
         0.8292189240455627, 0.5674595832824707, 0.8361266255378723, 0.8343343138694763, 0.0, 0.6598103046417236,
         0.5845543146133423, 0.0, 0.0, 0.6306214332580566, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.9865005612373352, 0.9815070033073424, 0.9038004279136658, 0.9120953679084778, 0.8330145478248596,
         0.8660752773284912, 0.6965143084526062, 0.0, 0.0, 0.5894452929496765, 0.0, 0.0, 0.679776132106781, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6653428673744202, 0.0, 0.0, 0.0, 0.0, 0.5730385780334473, 0.6123088002204895,
         0.0, 0.0, 0.0, 0.626666247844696, 0.0, 0.0, 0.9325371384620668, 0.0, 0.7041198015213013, 0.0,
         0.7817896008491516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

features = ['town', 'neighbourhood', 'street', 'road', 'alley', 'infrastructure', 'building', 'property', 'sky', 'lane',
            'city', 'urban area', 'residential area', 'house', 'facade', 'window', 'apartment', 'downtown', 'wall',
            'tours', 'roof', 'landmark', 'tourist attraction', 'structure', 'tree', 'metropolitan area', 'town square',
            'recreation', 'tower', 'street light', 'plaza', 'car', 'land vehicle', 'luxury vehicle', 'vehicle',
            'transport', 'family car', 'sedan', 'parking lot', 'metropolis', 'compact car', 'traffic', 'daytime',
            'city car', 'architecture', 'mixed use', 'suburb', 'traffic light', 'signaling device', 'light fixture',
            'pedestrian', 'evening', 'mid size car', 'parking', 'plant', 'skyscraper', 'palace', 'minivan',
            'executive car', 'cloud', 'estate', 'blue', 'volvo cars', 'subcompact car', 'classical architecture',
            'motor vehicle', 'toyota', 'sport utility vehicle', 'bmw', 'automotive design', 'winter', 'cityscape',
            'marketplace', 'market', 'home', 'real estate', 'mode of transport', 'public space', 'tourism',
            'road surface', 'compact mpv', 'church', 'cathedral', 'place of worship', 'steeple', 'spire', 'basilica',
            'château', 'medieval architecture', 'bell tower', 'historic site', 'stately home', 'history',
            'listed building', 'condominium', 'commercial building', 'hotel', 'reflection', 'mansion', 'historic house',
            'waterway', 'tower block', 'balcony', 'baptistery', 'signage', 'service', 'retail', 'advertising',
            'shopping', 'taxi', 'byzantine architecture', 'branch', 'monument', 'ancient history', 'sculpture',
            'column', 'statue', 'ancient roman architecture', 'ruins', 'woody plant', 'arch', 'ancient rome',
            'synagogue', 'triumphal arch']


# new_name = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123']

@app.route('/validate')
def validate():
    image = readFileContentAsImage('test_image.jpg')
    df = fromImageToDataFrame(image)
    return jsonify({'detected': make_validation(df)})


def fromImageToDataFrame(image):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds.json"
    client = vision.ImageAnnotatorClient()
    requests = [makeRequest(image)]
    googleVisionResponse = client.batch_annotate_images(requests)
    empty_df = pd.DataFrame(columns=features)
    data_df = fillDataFrameFromResonse(empty_df, googleVisionResponse)
    return data_df

def readFileContentAsImage(imagePath):
    with io.open(imagePath, 'rb') as image_file:
        content = image_file.read()
    return types.Image(content=content)

def makeRequest(image):
    return {
        'image': image,
        'features': [
            {'type': vision.enums.Feature.Type.LABEL_DETECTION, 'max_results': 30},
            {'type': vision.enums.Feature.Type.TEXT_DETECTION},
            {'type': vision.enums.Feature.Type.IMAGE_PROPERTIES}
        ],
    }


def fillDataFrameFromResonse(dataFrame, response):
    resp = response.responses[0]
    dic = {}
    for label in resp.label_annotations:
        if label.description in dataFrame:
            dic[label.description] = label.score
    return removeNaN(dataFrame.append(dic, ignore_index=True))


def removeNaN(dataFrame):
    return dataFrame.where(dataFrame.notna(), 0)


def make_validation(data_df):
    print(data_df)
    clf: XGBClassifier = load('clf.joblib')
    result = clf.predict(data_df).tolist()
    return result[0]


@app.route('/data')
def get_data():
    return jsonify(data)


if __name__ == '__main__':
    app.run()
