import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cv2
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import json
# -----------------------------
# PAGE
# -----------------------------

st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container{
padding-top:1rem;
padding-bottom:0rem;
padding-left:2rem;
padding-right:2rem;
}
</style>
""", unsafe_allow_html=True)

PRIMARY = "#2a9d8f"
SECONDARY = "#1f4e79"

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("drugs.csv")

df['renal_adjustment'] = df['renal_adjustment'].str.strip().str.lower()

df['renal_adjustment'] = df['renal_adjustment'].replace({
"yes":"Adjustment Required",
"no":"No Adjustment"
})

df['indication_group'] = df['indication'].apply(
lambda x: "Pneumonia" if "pneumonia" in x.lower() else "Other"
)

train_dir = "ML/CXR/CXR_dataset/train"
classes = os.listdir(train_dir)

# -----------------------------
# TITLE
# -----------------------------

st.markdown(
"""
<h1 style="
text-align:center;
color:#1f4e79;
margin-top:10px;
margin-bottom:20px;
">
Clinical Dosing & Pneumonia AI Dashboard
</h1>
""",
unsafe_allow_html=True
)

# -----------------------------
# METRICS
# -----------------------------

m1,m2,m3,m4 = st.columns(4)

m1.metric("Drugs", df['generic_name'].nunique())
m2.metric("Indications", df['indication'].nunique())
m3.metric("Administration", df['administration'].nunique())
m4.metric("Dose Adjustment", df['renal_adjustment'].nunique())

# -----------------------------
# ROW 1
# -----------------------------

c1,c2,c3 = st.columns([1,1,1.4])

# Indications
with c1:

    counts = df['indication_group'].value_counts().reset_index()
    counts.columns = ["type","count"]

    fig = px.pie(
        counts,
        names="type",
        values="count",
        color_discrete_sequence=[PRIMARY,SECONDARY],
        title="Indications"
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# Renal Adjustment
with c2:

    adj = df['renal_adjustment'].value_counts().reset_index()
    adj.columns = ["type","count"]

    fig = px.pie(
        adj,
        names="type",
        values="count",
        color_discrete_sequence=[SECONDARY,PRIMARY],
        title="Renal Adjustment"
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# Heatmap
with c3:

    table = pd.crosstab(df['indication'],df['renal_adjustment'])

    fig = px.imshow(
        table,
        color_continuous_scale="Teal",
        title="Indication vs Adjustment"
    )

    fig.update_layout(height=300)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ROW 2
# -----------------------------

counts = []

for c in classes:
    counts.append(len(os.listdir(os.path.join(train_dir,c))))

data = pd.DataFrame({
"class":classes,
"count":counts
})

c4,c5,c6 = st.columns(3)

# Training Distribution
with c4:

    fig = px.bar(
        data,
        x="class",
        y="count",
        color="class",
        color_discrete_sequence=[PRIMARY,SECONDARY],
        title="Training Distribution"
    )

    fig.update_layout(height=260,showlegend=False)

    st.plotly_chart(fig,use_container_width=True)

# Pneumonia vs Normal
with c5:

    fig = px.pie(
        data,
        names="class",
        values="count",
        color_discrete_sequence=[SECONDARY,PRIMARY],
        title="Pneumonia vs Normal"
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# Image Size Distribution
with c6:

    widths = []
    heights = []

    for c in classes:

        folder = os.path.join(train_dir,c)

        for img_name in os.listdir(folder)[:200]:

            img = cv2.imread(os.path.join(folder,img_name))

            heights.append(img.shape[0])
            widths.append(img.shape[1])

    size_df = pd.DataFrame({
        "Width": widths,
        "Height": heights
    })

    fig = px.histogram(
        size_df,
        x=["Width","Height"],
        nbins=30,
        title="Image Size Distribution",
        color_discrete_sequence=[PRIMARY,SECONDARY]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ROW 3
# -----------------------------

c7, c8 = st.columns(2)

# Pixel Intensity Distribution
with c7:

    sample_img_path = os.path.join( 
    train_dir,
    classes[0],
    random.choice(os.listdir(os.path.join(train_dir,classes[0])))
                )

    img = cv2.imread(sample_img_path,0)

    pixels = img.ravel()

    # تقليل عدد البكسلات
    pixels = np.random.choice(pixels, size=50000, replace=False)

    pixel_df = pd.DataFrame({"pixel":pixels})

    fig = px.histogram(
        pixel_df,
        x="pixel",
        nbins=80,
        title="Pixel Intensity Distribution",
        color_discrete_sequence=[PRIMARY]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)
# Dataset Distribution
with c8:

    base_path = "ML/CXR/CXR_dataset"
    splits = ["train","val","test"]

    data = []

    for split in splits:

        split_path = os.path.join(base_path, split)

        for cls in os.listdir(split_path):

            class_path = os.path.join(split_path, cls)

            count = len(os.listdir(class_path))

            data.append({
                "split": split,
                "class": cls,
                "count": count
            })

    dist_df = pd.DataFrame(data)

    fig = px.bar(
        dist_df,
        x="split",
        y="count",
        color="class",
        barmode="group",
        title="Dataset Distribution",
        color_discrete_sequence=[PRIMARY,SECONDARY]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ROW 4
# -----------------------------

c9, c10, c11 = st.columns(3)

epochs = list(range(1,11))

train_loss = [0.3735,0.1982,0.1809,0.1951,0.1810,0.1668,0.1470,0.1509,0.1578,0.1429]

val_acc = [0.8474,0.9077,0.8371,0.9248,0.9487,0.9339,0.9396,0.9442,0.9613,0.9374]

cm = np.array([[223,14],
               [20,621]])

# -----------------------------
# Training Loss
# -----------------------------

with c9:

    loss_df = pd.DataFrame({
        "Epoch":epochs,
        "Loss":train_loss
    })

    fig = px.line(
        loss_df,
        x="Epoch",
        y="Loss",
        markers=True,
        title="Training Loss",
        color_discrete_sequence=[PRIMARY]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)


# -----------------------------
# Validation Accuracy
# -----------------------------

with c10:

    acc_df = pd.DataFrame({
        "Epoch":epochs,
        "Accuracy":val_acc
    })

    fig = px.line(
        acc_df,
        x="Epoch",
        y="Accuracy",
        markers=True,
        title="Validation Accuracy",
        color_discrete_sequence=[SECONDARY]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)


# -----------------------------
# Confusion Matrix
# -----------------------------

with c11:

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal","Actual Pneumonia"],
        columns=["Pred Normal","Pred Pneumonia"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Teal",
        title="Confusion Matrix"
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ROW 5 - MODEL EVALUATION
# -----------------------------

c12, c13 = st.columns(2)

with open("model_metrics.json") as f:
    metrics = json.load(f)

roc_data = json.load(open("roc_data.json"))


# ROC Curve
with c12:

    roc_df = pd.DataFrame({
        "FPR": roc_data["fpr"],
        "TPR": roc_data["tpr"]
    })

    fig = px.line(
        roc_df,
        x="FPR",
        y="TPR",
        title=f"ROC Curve (AUC = {metrics['auc']:.2f})",
        color_discrete_sequence=[PRIMARY]
    )

    # الخط المتقطع (Random Classifier)
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=1, y1=1,
        line=dict(
            dash="dash",
            color="gray",
            width=2
        )
    )

    fig.update_layout(
        height=260,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig, use_container_width=True)
# Precision Recall F1
with c13:

    metrics_df = pd.DataFrame({
        "Metric":["Precision","Recall","F1 Score"],
        "Value":[
            metrics["precision"],
            metrics["recall"],
            metrics["f1"]
        ]
    })

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        color="Metric",
        title="Model Evaluation Metrics",
        color_discrete_sequence=[PRIMARY,SECONDARY,"#264653"]
    )

    fig.update_layout(height=260)

    st.plotly_chart(fig,use_container_width=True)