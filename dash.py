import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cv2
import random
import numpy as np
import json

st.set_page_config(layout="wide")

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

with open("model_metrics.json") as f:
    metrics = json.load(f)

roc_data = json.load(open("roc_data.json"))

# -----------------------------
# TITLE
# -----------------------------

st.markdown("""
<h1 style='text-align:center;color:#1f4e79'>
Clinical Dosing & Pneumonia AI Dashboard
</h1>
""",unsafe_allow_html=True)

# ======================================================
# X-RAY DATASET ANALYSIS
# ======================================================

st.markdown("## X-ray Dataset Analysis")

counts = []
for c in classes:
    counts.append(len(os.listdir(os.path.join(train_dir,c))))

data = pd.DataFrame({
"class":classes,
"count":counts
})

c1,c2,c3 = st.columns(3)

with c1:

    fig = px.bar(
        data,
        x="class",
        y="count",
        color="class",
        text="count",
        title="Training Distribution",
        color_discrete_sequence=[PRIMARY,SECONDARY]
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

with c2:

    fig = px.pie(
        data,
        names="class",
        values="count",
        title="Pneumonia vs Normal",
        color_discrete_sequence=[SECONDARY,PRIMARY]
    )

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

with c3:

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

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Pixel + dataset distribution
# -----------------------------

c4,c5 = st.columns(2)

with c4:

    sample_img_path = os.path.join(
        train_dir,
        classes[0],
        random.choice(os.listdir(os.path.join(train_dir,classes[0])))
    )

    img = cv2.imread(sample_img_path,0)

    pixels = img.ravel()
    pixels = np.random.choice(pixels, size=50000, replace=False)

    pixel_df = pd.DataFrame({"pixel":pixels})

    fig = px.histogram(
        pixel_df,
        x="pixel",
        nbins=80,
        title="Pixel Intensity Distribution",
        color_discrete_sequence=[PRIMARY]
    )

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

with c5:

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
        text="count",
        barmode="group",
        title="Dataset Distribution",
        color_discrete_sequence=[PRIMARY,SECONDARY]
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

# ======================================================
# TRAINING PERFORMANCE
# ======================================================

st.markdown("## Model Training Performance")

epochs = list(range(1,11))

train_loss = [0.3735,0.1982,0.1809,0.1951,0.1810,0.1668,0.1470,0.1509,0.1578,0.1429]
val_acc = [0.8474,0.9077,0.8371,0.9248,0.9487,0.9339,0.9396,0.9442,0.9613,0.9374]

cm = np.array([[223,14],[20,621]])

c6,c7,c8 = st.columns(3)

with c6:

    loss_df = pd.DataFrame({"Epoch":epochs,"Loss":train_loss})

    fig = px.line(
        loss_df,
        x="Epoch",
        y="Loss",
        markers=True,
        title="Training Loss",
        color_discrete_sequence=[PRIMARY]
    )

    fig.update_layout(height=340,yaxis_range=[0,0.5])

    st.plotly_chart(fig,use_container_width=True)

with c7:

    acc_df = pd.DataFrame({"Epoch":epochs,"Accuracy":val_acc})

    fig = px.line(
        acc_df,
        x="Epoch",
        y="Accuracy",
        markers=True,
        title="Validation Accuracy",
        color_discrete_sequence=[SECONDARY]
    )

    fig.update_layout(height=340,yaxis_range=[0,1])

    st.plotly_chart(fig,use_container_width=True)

with c8:

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

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

# ======================================================
# ROC + METRICS GRAPH
# ======================================================

c9,c10 = st.columns(2)

with c9:

    roc_df = pd.DataFrame({
        "FPR":roc_data["fpr"],
        "TPR":roc_data["tpr"]
    })

    fig = px.line(
        roc_df,
        x="FPR",
        y="TPR",
        title=f"ROC Curve (AUC = {metrics['auc']:.3f})",
        color_discrete_sequence=[PRIMARY]
    )

    fig.add_shape(
        type="line",
        x0=0,y0=0,
        x1=1,y1=1,
        line=dict(dash="dash",color="gray")
    )

    fig.update_layout(height=360)

    st.plotly_chart(fig,use_container_width=True)

with c10:

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
        text="Value",
        color="Metric",
        title="Model Evaluation Metrics",
        color_discrete_sequence=[PRIMARY,SECONDARY,"#264653"]
    )

    fig.update_traces(texttemplate='%{text:.2f}',textposition='outside')

    fig.update_layout(height=360,yaxis_range=[0,1])

    st.plotly_chart(fig,use_container_width=True)

# ======================================================
# FINAL MODEL PERFORMANCE
# ======================================================

st.markdown("## Final Model Performance")

m1,m2,m3,m4 = st.columns(4)

m1.metric("Accuracy",f"{metrics['accuracy']:.3f}")
m2.metric("Precision",f"{metrics['precision']:.3f}")
m3.metric("Recall",f"{metrics['recall']:.3f}")
m4.metric("F1 Score",f"{metrics['f1']:.3f}")

# ======================================================
# DRUG ANALYSIS
# ======================================================

st.markdown("## Drug Dosing Analysis")

m5,m6,m7,m8 = st.columns(4)

m5.metric("Drugs", df['generic_name'].nunique())
m6.metric("Indications", df['indication'].nunique())
m7.metric("Administration", df['administration'].nunique())
m8.metric("Dose Adjustment", df['renal_adjustment'].nunique())

c11,c12,c13 = st.columns([1,1,1.3])

with c11:

    counts = df['indication_group'].value_counts().reset_index()
    counts.columns = ["type","count"]

    fig = px.pie(
        counts,
        names="type",
        values="count",
        color_discrete_sequence=[PRIMARY,SECONDARY],
        title="Indications"
    )

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

with c12:

    adj = df['renal_adjustment'].value_counts().reset_index()
    adj.columns = ["type","count"]

    fig = px.pie(
        adj,
        names="type",
        values="count",
        color_discrete_sequence=[SECONDARY,PRIMARY],
        title="Renal Adjustment"
    )

    fig.update_layout(height=340)

    st.plotly_chart(fig,use_container_width=True)

with c13:

    table = pd.crosstab(df['indication'],df['renal_adjustment'])

    fig = px.imshow(
        table,
        text_auto=True,
        color_continuous_scale="Teal",
        title="Indication vs Adjustment"
    )

    fig.update_layout(height=360)

    st.plotly_chart(fig,use_container_width=True)