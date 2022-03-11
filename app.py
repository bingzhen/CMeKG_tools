from fastapi import FastAPI
import medical_ner
import model_re.medical_re as medical_re
import medical_cws
from enum import Enum

description = """
## Test sentence example

实体命名：抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。

关系抽取：据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷、2.人传人。

分词：抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。

"""

app = FastAPI(
    title="AIML NLP Model Warehouse",
    description=description,
    version="0.0.1"
)

class ModelName(str, Enum):
    ner = "医学实体命名"
    re = "医学关系抽取"
    cws = "医学分词"

@app.get("/models/{model_name}", tags=["模型API测试"])
async def get_model(model_name: ModelName, sentence: str):
    if model_name == ModelName.ner:
        my_pred = medical_ner.medical_ner()
        res = my_pred.predict_sentence(sentence)
        return res
        #抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。

    if model_name == ModelName.re:
        PATH_SCHEMA = "/home/bingzhen/Desktop/model_warehouse/CMeKG_tools/weights/medical_re/predicate.json"
        medical_re.load_schema(medical_re.config.PATH_SCHEMA)
        model4s, model4po = medical_re.load_model()
        res = medical_re.get_triples(sentence, model4s, model4po)
        return res
        # 据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷、2.人传人。

    if model_name == ModelName.cws:
        meg = medical_cws.medical_seg()
        res = meg.predict_sentence(sentence)
        return res
        #抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。
