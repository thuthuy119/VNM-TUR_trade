import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
import streamlit as st
import plotly.express as px

# cd "C:\Users\DELL\OneDrive\Máy tính\Dữ liệu XNK VN - TURKEY"
# streamlit run app.py

## IMPORT DỮ LIỆU VÀ TIỀN XỬ LÝ

st.set_page_config(page_title="PHÂN TÍCH DỮ LIỆU THƯƠNG MẠI GIỮA VIỆT NAM VÀ TURKEY", layout="wide")

st.title("Phần 1. Dữ liệu chi tiết theo mã HS")

# --- Đọc dữ liệu ---
# Bạn có thể giữ nguyên đường dẫn hiện tại hoặc thay bằng st.file_uploader nếu muốn.
excel_path = r'C:\Users\DELL\OneDrive\Máy tính\Dữ liệu XNK VN - TURKEY\Data trade VNM - TUR.xlsx'
df = pd.read_excel(excel_path, sheet_name='Data')

#----



#----------------------------------------------------------------

all_levels = ["HS2", "HS4", "HS6", "HS8"]
available_levels = []
for lvl in all_levels:
    if {lvl, f"{lvl}_product"}.issubset(df.columns):
        available_levels.append(lvl)
# Nếu file chỉ có đến HS6 thì radio sẽ không hiện HS8.
if not available_levels:
    st.error("Không tìm thấy các cột mã HS và tên sản phẩm (VD: HS2/HS2_product).")
    st.stop()

# --- Bước 1: Tạo selection bar HS2, HS4, HS6/HS8 ---
hs_level = st.selectbox("Chọn cấp độ mã HS:", available_levels)

# Map cột và độ dài mã để zfill
hs_code_col = hs_level
hs_name_col = f"{hs_level}_product"
hs_len_map = {"HS2": 2, "HS4": 4, "HS6": 6, "HS8": 8}
code_len = hs_len_map.get(hs_level, 2)

# Tạo cột hiển thị "Mã - Tên sản phẩm"
# Sửa lỗi cộng chuỗi với Series bằng cách ép kiểu CHO CẢ HAI phía + fillna
df["_hs_code_str"] = (
    df[hs_code_col]
    .astype(str)
    .str.replace(r"\.0+$", "", regex=True)  # nếu Excel đọc thành số float
    .str.zfill(code_len)
)
df["_hs_name_str"] = df[hs_name_col].astype(str).fillna("")
# Nếu tên là "nan" (do astype từ NaN), thay bằng chuỗi rỗng
df.loc[df["_hs_name_str"].str.lower() == "nan", "_hs_name_str"] = ""

df["HS_display"] = df["_hs_code_str"] + " - " + df["_hs_name_str"]
# Nếu không có tên thì chỉ hiện mã
df.loc[df["_hs_name_str"].eq(""), "HS_display"] = df["_hs_code_str"]

# Danh sách lựa chọn theo cấp đã chọn
opts = (
    df[[hs_code_col, "HS_display", "_hs_code_str"]]
    .drop_duplicates()
    .sort_values("_hs_code_str")
)
hs_selected_display = st.selectbox(f"Chọn mã {hs_level}:", opts["HS_display"].tolist())

# Lấy lại mã thật từ lựa chọn (trước dấu " - ")
hs_code = hs_selected_display.split(" - ")[0]

# Lọc theo mã đã chọn
df_filtered = df[df["_hs_code_str"] == hs_code].copy()

# Giới hạn năm 2020-2024
df_filtered = df_filtered[(df_filtered["Year"] >= 2020) & (df_filtered["Year"] <= 2024)]

# --- Bước 2: Chuẩn bị dữ liệu cho biểu đồ ---
criteria_list = [
    "Türkiye's imports from Viet Nam",
    "Viet Nam's exports to world",
    "Türkiye's imports from world",
]

df_plot = (
    df_filtered[df_filtered["Criteria"].isin(criteria_list)]
    .groupby(["Year", "Criteria"], as_index=False)["Value (000 USD)"]
    .sum()
    .sort_values(["Criteria", "Year"])
)

# --- Bước 3: Vẽ biểu đồ Plotly (line) ---
title_txt = f"So sánh giá trị thương mại {hs_selected_display} (2020–2024)"
fig = px.line(
    df_plot,
    x="Year",
    y="Value (000 USD)",
    color="Criteria",
    markers=True,
    title=title_txt
)
fig.update_layout(
    xaxis_title="Năm",
    yaxis_title="Giá trị (000 USD)",
    legend_title="Chỉ tiêu",
)

st.plotly_chart(fig, use_container_width=True)

# --- Bước 4: Tính bảng tỷ trọng ---
df_pivot = (
    df_plot.pivot_table(
        index="Year",
        columns="Criteria",
        values="Value (000 USD)",
        aggfunc="sum"
    )
    .reindex(range(2020, 2025))  # đảm bảo đủ 2020-2024 theo thứ tự
)

# Đổi tên cột cho ngắn gọn
col_tr_vn = "Türkiye's imports from Viet Nam"
col_vn_world = "Viet Nam's exports to world"
col_tr_world = "Türkiye's imports from world"

# Tránh chia cho 0
denom1 = df_pivot[col_tr_world].replace(0, np.nan)
denom2 = df_pivot[col_vn_world].replace(0, np.nan)

ratio1 = (df_pivot[col_tr_vn] / denom1) * 100
ratio2 = (df_pivot[col_tr_vn] / denom2) * 100

# Làm tròn 2 chữ số & format %
ratio1_fmt = ratio1.round(2).map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")
ratio2_fmt = ratio2.round(2).map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")

tbl = pd.DataFrame({
    "Year": df_pivot.index,
    "Tỷ trọng (VN→TR / TR nhập từ thế giới)": ratio1_fmt.values,
    "Tỷ trọng (VN→TR / VN xuất ra thế giới)": ratio2_fmt.values,
}).reset_index(drop=True)

st.subheader("Bảng tỷ trọng (%) – 2020–2024")
st.dataframe(tbl, use_container_width=True)

# (Không còn cần các cột tạm)
df.drop(columns=[c for c in ["_hs_code_str", "_hs_name_str"] if c in df.columns], inplace=True, errors="ignore")