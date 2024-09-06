"""Send a test message."""

import argparse
import json
import time

import requests

from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import get_conversation_template


def main():
    import random

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
    args.message = random.choice(questions)
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        # print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        # print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    # conv = get_conversation_template("qwen-7b-chat")
    conv = get_conv_template("qwen-7b-chat")

    from fastchat.prompt.custom_prompt import intent_slot_prompt

    conv.set_system_message(intent_slot_prompt)
    conv.append_message(conv.roles[0], args.message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "repetition_penalty": 1.5,
        "echo": False,  # echo参数用于将输入的prompt文本作为生成结果的一部分返回，这可以用于将输入的上下文与生成的文本结果组合在一起，增强文本的可读性和可解释性。
    }
    # print(f"==== request ====\n{gen_params}")
    start_time = time.time()
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    print(f"{conv.roles[0]}: {args.message}")
    print(f"{conv.roles[1]}: ", end="")
    prev = 0
    last_time = time.time()
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            cur_time = time.time()
            data = json.loads(chunk.decode())
            output = data["text"].strip()
            # print(output[prev:], end="", flush=True)
            # print(f"token/s={cur_time - last_time}")
            prev = len(output)
            last_time = cur_time

    print(f"{output=}")
    consume_time = time.time() - start_time
    print(f"time consuming: {consume_time}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--message",
        type=str,
        default="Human: 北京到上海的火车票有哪些？\n AI: 请问您希望选择哪种类型的车厢？",
    )
    args = parser.parse_args()

# while True:
    main()
