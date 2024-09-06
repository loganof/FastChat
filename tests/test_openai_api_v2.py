import os
from requests import request
import requests
from vllm import LLM

from tests.random_question import random_Q

prompt = "输入一段用户和AI助手对话，你需要从用户的对话中获得用户意图，并且提取出关键信息，而对于AI的回答则不需要提取。对话的意图以及对应的关键信息如下：\n        train_ticket_tool: 此意图为当用户询问关于查询火车票或订火车票相关的问题时，它会给予正确的回复。例如询问“我想看看后天到上海的火车票”，“12月1日北京到上海的火车票”等\n        train_ticket_tool的关键信息如下所示：\n            出发地（departure)\n            目的地 (destination)\n            出发日期（date)\n            车票类型 (type, 车次号或者对火车票的描述))\n            乘客姓名(name)\n            手机号(phone)\n            身份证号(ID_card)\n            你需要检查出发日期, 如果用户提供的日期比较模糊(比如周末、暑假、春运、**期间), 无法转换成具体的某一天，则不输出日期;如果可以转换成具体的某一天,则输出原始表述\n        例如：\n            “Human：我想查询一下下周四的火车票情况。\n            AI：请提供您的出发城市和目的地。\n            Human：从广州出发，目的地是北京。\n            AI：正在为您查询下周四从广州到北京的火车票，请稍候。\n            Human： 好的。\n            AI：查询到有7个火车票信息，请问你有什么偏好吗？\n            Human：高铁票二等座。\n            AI：好的，请提供您的姓名和身份证号。\n            Human：章三， 123456789876543210。\n            AI：确认一下，您要预定下周四从广州到北京的高铁二等座，对吗？\n            Human：是的，确认预定。“\n        需要返回的是{'target': 'train_ticket_tool', departure': '广州', 'destination': '北京', 'date': '下周四', 'type': '高铁二等座', 'name': '章三', 'ID_card': '123456789876543210'}\n    weather_query_tool: 此意图为当用户询问关于查询天气的问题时，它会给予正确的回复。例如询问“今天天气怎么样”，“今天多少度”等\n    weather_query_tool的关键信息如下：\n        地点(location)\n        日期(date)\n        你需要检查日期, 如果用户提供的日期比较模糊(比如周末、暑假、春运、**期间), 无法转换成具体的某一天，则不输出日期;如果可以转换成具体的某一天,则输出原始表述\n    例如：\n           “Human：我想查询一下下周四的天气情况。\n            AI：好的,您想查询哪里的天气呢。\n            Human：北京。“\n        你需要输出：\n            {'target': 'weather_query_tool', 'location': '北京', 'date': '下周四'}\n    airline_ticket_tool: 此模块为当用户询问关于查询机票或订机票相关的问题时，它会给予正确的回复。例如询问“我想看看后天到上海的机票”，“12月1日北京到上海的机票”等\n    airline_ticket_tool的关键信息如下：\n            出发地（departure)\n            目的地 (destination)\n            出发日期（date)\n            机票类型 (type, 航班号或者对机票的描述))\n            乘客姓名(name)\n            手机号(phone)\n            身份证号(ID_card)。\n        你需要检查出发日期, 如果用户提供的日期比较模糊(比如周末、暑假、春运、**期间), 无法转换成具体的某一天，则不输出日期;如果可以转换成具体的某一天,则输出原始表述。\n    例如：\n            Human：我想查询一下下周四的航班情况。\n            AI：请提供您的出发城市和目的地。\n            Human：从广州出发，目的地是北京。\n            AI：正在为您查询下周四从广州到北京的机票，请稍候。\n            Human： 好的。\n            AI：查询到有7个航班信息，请问你有什么偏好吗？\n            Human：最便宜的那一班吧。\n            AI：好的，请提供您的姓名和身份证号。\n            Human：章三， 123456789876543210。\n            AI：确认一下，您要预定下周四从广州到北京的最便宜的航班，对吗？\n            Human：是的，确认预定。“\n        你需要输出：\n            {'target': 'airline_ticket_tool', 'departure': '广州', 'destination': '北京', 'date': '下周四', 'type': '最便宜', 'name': '章三', 'ID_card': '123456789876543210'}\n        \n下面我给你一段对话，按照上面的要求，对这段对话提取出关键信息，对于AI的回答不需要提取, 没有出现的槽值不需要输出。以字典格式直接输出结果, 不需要输出说明。对话如下：\n" + random_Q

os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
param = {
    "model": "qwen2_0.5b_3typeslotintent_sft_20240815e3",
    "messages": [
        {
            "content": prompt,
            "role": "user"
        }
    ],
    "temperature": 1.0,
    "max_tokens": 300,
    "repetition_penalty": 1.5,
    "stop": "<|im_end|>"
}

# 设置请求URL
openai_api_url_vllm= "http://127.0.0.1:12001/v1/chat/completions"
openai_api_url_lightllm = "http://127.0.0.1:12001/v1/chat/completions"


# 设置请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": "Empty",  # 如果不需要API_KEY可以删除这一行
}

def test_openai_api_vllm():
    response = requests.post(openai_api_url_vllm, headers=headers, json=param)
    # print(response.json().get("choices")[0].get("text"))
    print(response.json())


def test_openai_api_lightllm():
    response = requests.post(openai_api_url_lightllm, headers=headers, json=param)
    # print(response.json().get("choices")[0].get("text"))
    print(response.json())