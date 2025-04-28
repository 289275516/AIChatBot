import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import PtEngine, RequestConfig, InferRequest
model = 'Qwen/Qwen2.5-0.5B-Instruct'

# 加载推理引擎
engine = PtEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

content = input("请输入你的问题：")

# 这里使用了2个infer_request来展示batch推理
infer_requests = [InferRequest(messages=[{'role': 'user', 'content': content}])]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
