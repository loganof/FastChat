"""Benchmarking script to test the throughput of serving workers."""

import argparse
import json

import requests
import threading
import time

from fastchat.conversation import get_conv_template


def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": args.model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = get_conv_template("qwen-7b-chat")
    from fastchat.prompt.custom_prompt import intent_slot_prompt

    conv.set_system_message(intent_slot_prompt)
    conv.append_message(conv.roles[0], "查询明天的天气")
    prompt_template = conv.get_prompt()
    prompts = [prompt_template for _ in range(args.n_thread)]
    headers = {"User-Agent": "fastchat Client"}
    ploads = [
        {
            "model": args.model_name,
            "prompt": prompts[i],
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.7,
            # "stop": conv.sep,
            # "echo": False
        }
        for i in range(len(prompts))
    ]

    def send_request(results, i):
        if args.test_dispatch:
            ret = requests.post(
                controller_addr + "/get_worker_address", json={"model": args.model_name}
            )
            thread_worker_addr = ret.json()["address"]
        else:
            thread_worker_addr = worker_addr
        print(f"thread {i} goes to {thread_worker_addr}")
        response = requests.post(
            thread_worker_addr + "/worker_generate_stream",
            headers=headers,
            json=ploads[i],
            stream=True,
        )
        # k = list(
        #     response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0")
        # )
        # response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
        # error_code = json.loads(k[-2].decode("utf-8"))["error_code"]
        # print(f"=== Thread {i} ===, words: {1}, error code: {error_code}")
        previous_text = ""
        start_time = time.time()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                decoded_line = json.loads(chunk.decode("utf-8"))
                current_text = decoded_line["text"].strip()
                error_code = decoded_line["error_code"]

                # 计算新生成的单词数量
                # new_words = len(current_text.split(" ")) - len(previous_text.split(" "))
                # if new_words > 0:
                #     elapsed_time = time.time() - start_time
                #     print(f"Generated {new_words} new word(s) in {elapsed_time:.5f} seconds.")
                # start_time = time.time()  # 重置计时器

            previous_text = current_text
        response_new_words = previous_text
        print(f"output: {response_new_words}")
        # results[i] = len(response_new_words.split(" ")) - len(prompts[i].split(" "))
        results[i] = len(response_new_words.split(" "))

    # use N threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        # time.sleep(0.5)
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Time (POST): {time.time() - tik} s")
    # n_words = 0
    # for i, response in enumerate(results):
    #     # print(prompt[i].replace(conv.sep, "\n"), end="")
    #     # make sure the streaming finishes at EOS or stopping criteria
    #     k = list(response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
    #     response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
    #     # print(response_new_words)
    #     n_words += len(response_new_words.split(" ")) - len(prompts[i].split(" "))
    n_words = sum(results)
    time_seconds = time.time() - tik
    print(
        f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
        f"throughput: {n_words / time_seconds} words/s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="vicuna")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--n-thread", type=int, default=8)
    parser.add_argument("--test-dispatch", action="store_true")
    args = parser.parse_args()
    while True:
        main()
        time.sleep(0.01)
