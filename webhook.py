import json
import os
import traceback

import boto3
import spacy
from flask import Flask, request

from src.info_extractor import InfoExtractor
from src.model import RatingModel

spacy.load('en_core_web_sm')


app = Flask(__name__)


@app.route("/")
def test():
    return {"keshav":"Keshav is great"}
@app.route("/default",methods=['POST'])
def parser_api():
    print('api started')
    try:
        if request.method == "POST":
            # print(resume1)
            x=request.json
            resume1=x['resume']
            print(resume1)
            BUCKET_NAME = 'highporesume'
            LOCAL_FILE_NAME = 's3resumeTest1.pdf'
            s3 = boto3.client('s3', aws_access_key_id='xxxxxxxxxx', aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            s3.download_file(BUCKET_NAME, resume1, LOCAL_FILE_NAME)

            path_to_resume = LOCAL_FILE_NAME
            # --------------------------------
            _type='lda'
            model_path = "C:/Users/techm/PycharmProjects/Resume-Rater/src/models/model_lda/model.json"
            r = RatingModel(_type, model_path)
            infoExtractor = InfoExtractor(r.nlp, r.parser)
            data2=r.test(path_to_resume, infoExtractor)
            app_json2 = json.dumps(data2)
            print(app_json2)
            #---------------------------------

            os.remove(LOCAL_FILE_NAME)
            return app_json2


    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return "Oops... 😮 I am not able to help you at the moment, please try again.."




if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port {}".format(port))
    app.run(debug=True)






