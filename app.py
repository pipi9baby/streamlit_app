import streamlit as st
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizerFast,
)
import torch
import plotly.express as px

@st.cache_resource
def load_bert_model(model_version, model_dir, classes_num, device):
    tokenizer = BertTokenizerFast.from_pretrained(model_version, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=classes_num
        ).to(device)
    model.eval()
    return tokenizer, model


model_dir = r"C:\Users\minhan\Documents\ch_sentence_classification\results\best_weight" # best_weight 路徑
classes_names = ['其他','來電訊息','政府訊息','垃圾廣告','認證碼簡訊','包裹通知','帳單訊息'] # 類別列表(依照CSV檔案的順序)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("A Simple Streamlit Web App")

model_version = "bert-base-chinese"
tokenizer, model = load_bert_model(model_version, model_dir, len(classes_names), device)

with st.form("my_form"):
    text = st.text_area("Enter the text you want to inference!!", "", height=200)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        with torch.no_grad():
            text = tokenizer(text)
            text = torch.tensor(text["input_ids"]).unsqueeze(0).to(device)
            result = torch.nn.functional.softmax(model(text).logits)[0].tolist()
            result = [ item*100 for item in result]

        df = pd.DataFrame({"values":result, "class_name":classes_names})
        fig = px.pie(df, values='values', names='class_name', title='Prediction results')
        fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig, use_container_width=True)