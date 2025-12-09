# MultiModal LLM Test
## イントロ
このリポジトリは[【ChatGPT】マルチモーダルRAGのリファレンス実装 ～多様な情報源から一貫性のある結果を引き出す～](https://qiita.com/ksonoda/items/28586434904c26ec465b)のサンプルコードを動かしたものです. 

およびその元ネタと思われる
[Advanced Multi-Modal Retrieval using GPT4V and Multi-Modal Index/Retriever](https://docs.llamaindex.ai/en/stable/examples/multi_modal/gpt4v_multi_modal_retrieval/)
を参照しています


関連すると思われる日本語サイトのリスト
- [LlamaIndexのマルチモーダルを試す](https://zenn.dev/kun432/scraps/308e9750c822bc)
- [LlamaIndex の マルチモーダルRAG のしくみ](https://note.com/npaka/n/n53e8aabed0f2)

マルチモーダルRAGに関するドキュメントのリスト
- [Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V](https://docs.llamaindex.ai/en/stable/examples/multi_modal/image_to_image_retrieval/#retrieve-images-from-multi-modal-index-given-the-image-query)

## 準備
### 環境変数
API keyをセットします
```
touch .env
```
でファイルを生成して
```
OPENAI_API_KEY = "Your OpenAI API key"
```
を書き加えてください. 

### Python
- Python=3.10
- pip install -r requirements.txt

(Anacodaによる構築がおすすめ。例えば、
```
conda create -n mllm python=3.10

conda activate mllm

pip install -r requirements.txt)
```

