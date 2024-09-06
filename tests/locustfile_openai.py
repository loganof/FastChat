import json
import logging
import random
import time
import uuid

from fastchat.conversation import get_conv_template
from locust import HttpUser, task


class HelloWorldUser(HttpUser):
    # wait_time = between(0,0.2)

    @task
    def query_resolution(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",  # 添加 Accept 头，明确告诉服务器返回 JSON 数据
        }

        from fastchat.prompt.custom_prompt import intent_slot_prompt
        questions = [
            "Human: 北京到上海的火车票有哪些？\n AI: 请问您希望选择哪种类型的车厢？",
            "Human: 今天的天气怎么样？\n AI: 您是想知道哪个城市的天气呢？",
            "Human: 最近有什么好看的电影推荐吗？\n AI: 您喜欢什么类型的电影？",
            "Human: 哪些餐厅提供外卖服务？\n AI: 请问您想知道哪个城市或地区的餐厅信息？",
            "Human: 最近的新闻头条是什么？\n AI: 您是想了解国内还是国际的新闻？",
            "Human: 现在的股市行情怎么样？\n AI: 您关心的是哪个市场的股票呢？",
            "Human: 如何准备一个好的演讲？\n AI: 请问演讲的主题是什么？",
            "Human: 目前有哪些热门的旅游景点？\n AI: 请问您对哪类景点比较感兴趣？",
            "Human: 如何选择一款合适的手机？\n AI: 您更关注的是手机的哪个方面？",
            "Human: 学习Python的最佳方法是什么？\n AI: 您目前的编程经验如何？",
        ]
        question = random.choice(questions)

        param = {
            "model": "/data/laiqinglong/model/Qwen2-7B-Instruct",
            "messages": [
                {
                    "content": intent_slot_prompt + f"/n{question}",
                    "role": "user"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "repetition_penalty": 1.5,
            "stop": "<|im_end|>"
        }
        start = time.time()
        r = self.client.post(f"/v1/chat/completions", headers=headers, json=param)
        end = time.time()
        print(f"{r.json()}")
        duration = end - start
        logging.getLogger().info(f"time consuming {duration}s")
