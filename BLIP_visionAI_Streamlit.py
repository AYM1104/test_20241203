import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from google.cloud import vision
import io
import os

# Google Cloud認証ファイル（JSONキー）へのパスを設定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ayu/Tech0/Step2/WebApp/Team/JSON/ayu1104-90d5ce490d67.json"


# BLIPモデルとプロセッサをロード
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Google Vision APIクライアントを初期化
@st.cache_resource
def load_vision_client():
    return vision.ImageAnnotatorClient()

vision_client = load_vision_client()

# ーーーーーーーーーーーーーーーーーーーーーーー〈関数を作成〉ーーーーーーーーーーーーーーーーーーーーーーーーーーーー

# 画像から説明文を生成する関数（BLIP）
def describe_image(image):
    """画像からキャプションを生成"""
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Google Vision APIでラベルを検出する関数
def analyze_image_labels(image):
    """
    指定された画像をGoogle Vision APIで解析し、ラベルを返す関数。

    Args:
        image (PIL.Image.Image): PIL形式の画像。

    Returns:
        list: ラベルとスコアのリスト。
    """
    # Pillow画像をバイト形式に変換
    with io.BytesIO() as image_bytes:
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        image_content = image_bytes.read()

    # Google Vision APIに渡す画像オブジェクトを作成
    vision_image = vision.Image(content=image_content)

    # ラベル検出（Google Vision API）
    response = vision_client.label_detection(image=vision_image)
    labels = response.label_annotations

    # 結果をリスト形式に変換
    labels_list = [
        {"description": label.description, "score": label.score}
        for label in labels
    ]
    return labels_list

# ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

# Streamlitアプリ
st.title("画像解析アプリ")
st.write("画像をアップロードして、その内容を説明します！")

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # アップロードされた画像を読み込み
    image = Image.open(uploaded_file).convert("RGB")

    # アップロードされた画像を表示（サイズ調整）
    st.header("アップロードされた画像")
    st.image(image, caption="アップロードされた画像", use_container_width=False, width=400)

    # 画像解析結果を表示
    st.header("解析結果")
    
    # BLIPによるキャプション生成
    with st.spinner("画像のキャプションを生成中..."):
        caption = describe_image(image)
    st.success("キャプション生成完了!")
    st.write(f"**画像の説明:** {caption}")

    # Google Vision APIによるラベル検出
    st.header("Google Vision APIのラベル検出結果")
    with st.spinner("画像を解析中..."):
        labels = analyze_image_labels(image)
    st.success("ラベル解析完了!")
    
    # ラベル結果を表示
    if labels:
        st.write("以下のラベルが検出されました:")
        for label in labels:
            st.write(f"- {label['description']} (信頼度: {label['score']:.2f})")
    else:
        st.write("ラベルが検出されませんでした。")