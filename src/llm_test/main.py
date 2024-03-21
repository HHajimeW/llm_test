import os
import time
import json
import re

import pandas as pd
from groq import Groq
import anthropic
from openai import OpenAI

from pydantic import BaseModel, ValidationError

def check_json_schema(data:str):
    try: 
        data = json.loads(data)
    except ValueError as e:
        return ['NG',data,[e]]
    # 期待されるスキーマの定義
    expected_schema = {
        'keywords': str,
    }

    # エラーを記録するリスト
    errors = []

    # スキーマの各キーに対してチェックを行う
    for key, expected_type in expected_schema.items():
        if key not in data:
            errors.append(f"Key '{key}' is missing.")

    if len(errors) == 0:
        return ['OK',data,""]
    else:
        return ['NG',data,errors]

instruction = """Please extract the technical keywords in Japanese from the following text. 

私は健康科学の分野でニューラルネットワークによる医療費の予測というテーマで研究を行っています。この研究を一言で表すと、医療費の予測を機械学習ですることで予防医療や重視すべき治療、方針を決定する際の支えとすることです。 この研究テーマの背景として、日本が抱えている大きな問題である高額医療費があります。将来的には私の研究分野の蓄積により医療費の増加を抑え、より必要とされている社会保障をはじめとした他の分野に国の財源を回すことができる可能性があります。 今後もし研究を続けるなら、社会的に意義があり、人々の生活に関わる内容をしたいと考えております。
"""


synonym_japanese_instruction = """機械学習の類義語で具体的な技術キーワードを10つ日本語で提案し、keywordsというキーをもつjson形式でそのキーワードのみ出力してください。
"""

japanese_instruction = """以下の文章から技術的な専門用語を日本語で抽出し、keywordというキーだけを持つjson形式でキーワードのjsonのみ出力してください。
音声認識 (音響/言語モデル適応、End-to-End、ダイアライゼーション、音声強調/分離、音声感情認識、Kaldi/ESPnet活用など)、音響認識 (異常音検知、音響シーン分類、音響イベント検出、キャプション生成など)、時系列信号処理と機械学習 (スパースモデリング、信号復元、状態推定/予測のための機械学習など) のいずれかの研究開発を担当いただきながら、チームメンバーや後進の研究開発の指導を行っていただくことも期待します。いずれは研究チームを引っ張るリーダーとなっていただき、音声/音響/時系列信号処理技術の研究戦略検討や新事業の構想なども担っていただく人財となることを期待します。特に、音声認識に関しては、音声認識業界の経験と最新技術の知識をもとに、新事業（あるいは新研究テーマ）を構想することや、音声認識システムの開発経験と顧客需要の把握にもとづき、最新技術を取り入れた音声認識システムを提案することを期待します
"""

def extract_keywords_with_groq():
    groq_client = Groq(
        api_key="YOUR_GROQ_API_KEY",
    )

    start_time = time.time()
        
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": japanese_instruction,
            }
        ],
        # model="llama2-70b-4096",
        model="gemma-7b-it",
        temperature=0.0,
    )
               
    end_time = time.time()

    diff = end_time - start_time

    text = chat_completion.choices[0].message.content
    json_str_match = re.search(r'{.*?}', text, re.DOTALL)
    json_str = json_str_match.group(0)
    
    return json_str, diff


claude_synonym_japanese_instruction = """機械学習の類義語で具体的な技術キーワードを日本語で提案し、keywordsというキーをもつjson形式でそのキーワードのみ出力してください。"""


claude_japanese_instruction = """以下から技術キーワード(keywords)を日本語で抽出してください。出力は[keywords]をキーとしてもつjson形式でお願いします。
音声認識 (音響/言語モデル適応、End-to-End、ダイアライゼーション、音声強調/分離、音声感情認識、Kaldi/ESPnet活用など)、音響認識 (異常音検知、音響シーン分類、音響イベント検出、キャプション生成など)、時系列信号処理と機械学習 (スパースモデリング、信号復元、状態推定/予測のための機械学習など) のいずれかの研究開発を担当いただきながら、チームメンバーや後進の研究開発の指導を行っていただくことも期待します。いずれは研究チームを引っ張るリーダーとなっていただき、音声/音響/時系列信号処理技術の研究戦略検討や新事業の構想なども担っていただく人財となることを期待します。特に、音声認識に関しては、音声認識業界の経験と最新技術の知識をもとに、新事業（あるいは新研究テーマ）を構想することや、音声認識システムの開発経験と顧客需要の把握にもとづき、最新技術を取り入れた音声認識システムを提案することを期待します
"""

test_prompt = """機械学習について教えてください
"""

def extract_keywords_with_claude():
    claude_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="YOUR_ANTHROPIC_API_KEY",
    )

    start_time = time.time()
    message = claude_client.messages.create(
        # model="claude-3-sonnet-20240229",
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.0,
        system="You are an excellent researcher. You are well-versed in your field of expertise.",
        messages=[
            {"role": "user", "content": claude_japanese_instruction},
            {"role": "assistant","content": "{"},
        ]
    )
    
    end_time = time.time()
    
    diff = end_time - start_time

    text = "{" + message.content[0].text
    
    return text, diff
    
gpt_japanese_system_instruction = """
You are an excellent researcher. You are well-versed in your field of expertise.

# Expected JSON response
{"keywords":[keyword1, keyword2, ..., keywordN]}
"""

gpt_japanese_instruction = """以下から技術キーワード(keywords)を日本語でjson形式で抽出してください。
音声認識 (音響/言語モデル適応、End-to-End、ダイアライゼーション、音声強調/分離、音声感情認識、Kaldi/ESPnet活用など)、音響認識 (異常音検知、音響シーン分類、音響イベント検出、キャプション生成など)、時系列信号処理と機械学習 (スパースモデリング、信号復元、状態推定/予測のための機械学習など) のいずれかの研究開発を担当いただきながら、チームメンバーや後進の研究開発の指導を行っていただくことも期待します。いずれは研究チームを引っ張るリーダーとなっていただき、音声/音響/時系列信号処理技術の研究戦略検討や新事業の構想なども担っていただく人財となることを期待します。特に、音声認識に関しては、音声認識業界の経験と最新技術の知識をもとに、新事業（あるいは新研究テーマ）を構想することや、音声認識システムの開発経験と顧客需要の把握にもとづき、最新技術を取り入れた音声認識システムを提案することを期待します
"""

def extract_keywords_with_gpt():
    openai_client = OpenAI(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="YOUR_OPENAI_API_KEY",
    )

    start_time = time.time()

    completion = openai_client.chat.completions.create(
        # model="ft:gpt-3.5-turbo-0613:labbase:info-manual-data:7vS79ZFi",
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": gpt_japanese_system_instruction},
            {"role": "user", "content": claude_synonym_japanese_instruction}
        ],
        max_tokens=1000,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    # completion = openai_client.chat.completions.create(
    #     # model="ft:gpt-3.5-turbo-0613:labbase:info-manual-data:7vS79ZFi",
    #     model="gpt-3.5-turbo-0125",
    #     messages=[
    #         {"role": "system", "content": gpt_japanese_system_instruction},
    #         {"role": "user", "content": gpt_japanese_instruction}
    #     ],
    #     max_tokens=1000,
    #     temperature=0.0,
    #     functions=[
    #         {
    #             "name": "extract_keywords",
    #             "description": "文章の中から技術的なキーワードを抽出します。",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "keywords": {
    #                         "type": "array",
    #                         "items": {"type": "string"},
    #                         "description": "技術キーワードリスト"
    #                     },
    #                 },
    #                 "required": ["keywords"]
    #             }
    #         }
    # ],
    # function_call="auto",
    # )
    
    end_time = time.time()

    diff = end_time - start_time

    # res = completion.choices[0].message.function_call.arguments
    res = completion.choices[0].message.content
    return res, diff

if __name__ == "__main__":
    results = []

    for i in range(100):
        # response = extract_keywords_with_groq()
        # response = extract_keywords_with_claude()
        response = extract_keywords_with_gpt()
        result = check_json_schema(response[0])
        result.append(response[1])
        results.append(result)
        print(result)
        
        time.sleep(2)
        
    pandas_data = pd.DataFrame(results,columns=['result','data','error','time'])
    print(pandas_data)    
    
    pandas_data.to_csv('data/synonym/gpt3_5.csv', index=False)
    
    # for i in range(100):
    #     extract_keywords_with_gpt()