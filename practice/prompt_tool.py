import base64
import json
import os
from io import BytesIO

import requests
from PIL import Image
from pydantic import BaseModel, Field

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import LLMMathChain, SerpAPIWrapper


def generate_image(prompt: str) -> str:
    """
    根据提示词生成对应的图片

    Args:
        prompt (str): 英文提示词

    Returns:
        str: 图片的路径
    """
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "negative_prompt": "(worst quality:2), (low quality:2),disfigured, ugly, old, wrong finger",
        "steps": 20,
        "sampler_index": "Euler a",
        "sd_model_checkpoint": "cheeseDaddys_35.safetensors [98084dd1db]",
        # "sd_model_checkpoint": "anything-v3-fp16-pruned.safetensors [d1facd9a2b]",
        "batch_size": 1,
        "restore_faces": True
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        images = response_data['images']

        for index, image_data in enumerate(images):
            img_data = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_data))
            file_name = f"image_{index}.png"
            file_path = os.path.join(os.getcwd(), file_name)
            img.save(file_path)
            print(f"Generated image saved at {file_path}")
            return file_path
    else:
        print(f"Request failed with status code {response.status_code}")


def random_poem(arg: str) -> str:
    """
    随机返回中文的诗词

    Returns:
        str: 随机的中文诗词
    """
    llm = OpenAI(temperature=0.9)
    text = """
        能否帮我从中国的诗词数据库中随机挑选一首诗给我，希望是有风景，有画面的诗：
        比如：山重水复疑无路，柳暗花明又一村。
    """
    return llm(text)


def prompt_generate(idea: str) -> str:
    """
    生成图片需要对应的英文提示词

    Args:
        idea (str): 中文提示词

    Returns:
        str: 英文提示词
    """
    llm = OpenAI(temperature=0, max_tokens=2048)
    res = llm(f"""
    Stable Diffusion is an AI art generation model similar to DALLE-2.
    Below is a list of prompts that can be used to generate images with Stable Diffusion:

    - portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski
    - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson
    - ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image
    - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski
    - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt
    - athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration
    - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo
    - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha

    I want you to write me a list of detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more.

    IDEA: {idea}""")
    return res


class PromptGenerateInput(BaseModel):
    """
    生成英文提示词所需的输入模型类
    """
    idea: str = Field()


class GenerateImageInput(BaseModel):
    """
    生成图片所需的输入模型类
    """
    prompt: str = Field(description="英文提示词")


tools = [
    Tool.from_function(
        func=random_poem,
        name="诗歌获取",
        description="随机返回中文的诗词"
    ),
    Tool.from_function(
        func=prompt_generate,
        name="提示词生成",
        description="生成图片需要对应的英文提示词，当前工具可以将输入转换为英文提示词，以便方便生成",
        args_schema=PromptGenerateInput
    ),
    Tool.from_function(
        func=generate_image,
        name="图片生成",
        description="根据提示词生成对应的图片，提示词需要是英文的，返回是图片的路径",
        args_schema=GenerateImageInput
    ),
]


def main():
    """
    主函数，初始化代理并执行对话
    """
    llm = OpenAI(temperature=0)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("帮我生成一张诗词的图片?")


if __name__ == '__main__':
    main()
