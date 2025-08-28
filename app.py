import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests, io



# cd "C:\Users\DELL\OneDrive\Máy tính\Dữ liệu XNK VN - TURKEY"
# streamlit run app.py

## IMPORT DỮ LIỆU VÀ TIỀN XỬ LÝ

st.set_page_config(layout="wide")

st.title("PHÂN TÍCH DỮ LIỆU THƯƠNG MẠI GIỮA VIỆT NAM VÀ TURKEY")



# --- Đọc dữ liệu ---
# Bạn có thể giữ nguyên đường dẫn hiện tại hoặc thay bằng st.file_uploader nếu muốn.
url = "https://raw.githubusercontent.com/thuthuy119/VNM-TUR_trade/main/Data.xlsx"
content = requests.get(url).content
df = pd.read_excel(io.BytesIO(content), sheet_name="Data", engine="openpyxl")


df["HS2"] = df["HS2"].astype(str).str.replace(r"\D", "", regex=True).str[:2].str.zfill(2)
df["HS4"] = df["HS4"].astype(str).str.replace(r"\D", "", regex=True).str[:4].str.zfill(4)
df["HS6"] = df["HS6"].astype(str).str.replace(r"\D", "", regex=True).str[:6].str.zfill(6)


# ================== HELPERS GỌN ==================
st.header("Phần 1. Dữ liệu xuất nhập khẩu 5 năm (2020 - 2024) (Nguồn: Trademap)")


CRIT = "Turkey nhập khẩu từ Việt Nam"
@st.cache_data(show_spinner=False)
def prep_df(_df: pd.DataFrame, crit: str) -> pd.DataFrame:
    d = _df.copy()
    d["Criteria"] = d["Criteria"].astype(str).str.strip()
    d = d[d["Criteria"].eq(crit)]
    if d.empty: return d
    d["HS2"] = d["HS2"].astype(str).str.replace(r"\.0$","", regex=True).str.zfill(2)
    for lv in ("HS4","HS6"):
        if lv in d: d[lv] = d[lv].astype(str).str.replace(r"\.0$","", regex=True)
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Value (000 USD)"] = pd.to_numeric(d["Value (000 USD)"], errors="coerce").fillna(0.0)
    return d

@st.cache_data(show_spinner=False)
def agg_level(df, level, hs2_code=None):
    if hs2_code and level in ("HS4","HS6"):
        df = df[df["HS2"] == str(hs2_code).zfill(2)]          # narrow theo MÃ HS2
    lab = df[level].astype(str) + " - " + df[f"{level}_product"].astype(str)
    g = (pd.DataFrame({"label": lab, "Year": df["Year"], "Value (000 USD)": df["Value (000 USD)"]})
         .groupby(["label","Year"], as_index=False)["Value (000 USD)"].sum())
    years = sorted(g["Year"].dropna().astype(int).unique().tolist())
    return g, years

@st.cache_data(show_spinner=False)
def donut_parts(agg, year:int, topN:int):
    v = agg[agg["Year"]==int(year)][["label","Value (000 USD)"]]
    total = float(v["Value (000 USD)"].sum()) if not v.empty else 0.0
    if v.empty or total <= 0:
        return pd.DataFrame(columns=["label","Value (000 USD)"]), [], 0.0
    top = v.nlargest(topN, "Value (000 USD)")
    others = total - float(top["Value (000 USD)"].sum())
    if others > 0:
        pie = pd.concat([top, pd.DataFrame({"label":["Khác"], "Value (000 USD)":[others]})], ignore_index=True)
        order = top["label"].tolist() + ["Khác"]
    else:
        pie, order = top.copy(), top["label"].tolist()
    return pie, order, total

def cmap_from(oa, ob):
    pal = px.colors.qualitative.Set2 + px.colors.qualitative.Set3 + px.colors.qualitative.Plotly
    labs = [l for l in dict.fromkeys([*(oa or []), *(ob or [])]) if l!="Khác"]
    m = {lab: pal[i%len(pal)] for i,lab in enumerate(labs)}; m["Khác"]="#A0A0A0"; return m

def plot_donut(pie, order, year, total, cmap, hide=True):
    if pie.empty:
        st.info(f"Không có dữ liệu (tổng = 0) cho năm {year}."); return
    fig = px.pie(pie, names="label", values="Value (000 USD)", hole=0.55,
                 category_orders={"label":order}, color="label", color_discrete_map=cmap)
    fig.update_traces(sort=False, textposition="inside", texttemplate="%{percent:.1%}",
                      hovertemplate="<b>%{label}</b><br>Giá trị: %{value:,.0f} (000 USD)"
                                    "<br>Tỷ trọng: %{percent:.2%}<extra></extra>")
    fig.update_layout(showlegend=not hide, legend=dict(orientation="h", x=0, y=-0.2, font=dict(size=11)),
                      margin=dict(t=10,l=10,r=10,b=90), height=520,
                      annotations=[dict(text=f"<b>{int(year)}</b><br>{total:,.0f}", x=0.5, y=0.5, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

# ================== DỮ LIỆU CHUNG ==================

dfp = prep_df(df, CRIT)
if dfp.empty: st.warning("Không có dữ liệu cho chỉ tiêu đã chọn."); st.stop()

# ================== (A) HS2: CHỌN NĂM & VẼ TRƯỚC ==================
agg2, yrs2 = agg_level(dfp, "HS2")

# chọn năm từ HS2 (chưa cần biết HS2 cụ thể/HS4-6)
if not yrs2: st.warning("Không có năm nào ở cấp HS2."); st.stop()
default_years = yrs2[-2:] if len(yrs2)>=2 else yrs2
years_sel = st.multiselect("Chọn **2 năm** so sánh (áp dụng cho cả HS2 & HS4/HS6):",
                           yrs2, default=default_years, key="years_duo")
if len(years_sel)!=2: st.stop()
y1, y2 = sorted(map(int, years_sel))

c3,c4 = st.columns([2,1])
with c3:  topN = st.radio("Top N:", [10,20,30,50], horizontal=True, key="topn_duo")
with c4:  hide_legend = st.checkbox("Ẩn legend", True, key="hide_legend_duo")

# vẽ 2 donut HS2
p1_2,o1_2,t1_2 = donut_parts(agg2, y1, topN)
p2_2,o2_2,t2_2 = donut_parts(agg2, y2, topN)
cmap2 = cmap_from(o1_2,o2_2)

st.subheader("1.1. Top các mặt hàng Việt Nam xuất khẩu sang Turkey nhiều nhất")

st.markdown(f'###### Tất cả mã HS cấp 2')
a,b = st.columns(2)
with a: st.caption(f"Cơ cấu HS2 – {y1} (Top {topN})"); plot_donut(p1_2,o1_2,y1,t1_2,cmap2,hide_legend)
with b: st.caption(f"Cơ cấu HS2 – {y2} (Top {topN})"); plot_donut(p2_2,o2_2,y2,t2_2,cmap2,hide_legend)

# ================== (B) ĐẨY 2 SELECTION XUỐNG ĐÂY ==================


# HS2 options (chỉ những HS2 có tổng > 0)
hs2_opt = (
    dfp.groupby(["HS2","HS2_product"], as_index=False)["Value (000 USD)"].sum()
)
hs2_opt = hs2_opt[hs2_opt["Value (000 USD)"] > 0]
hs2_opt["hs2_label"] = hs2_opt["HS2"] + " - " + hs2_opt["HS2_product"].astype(str)
hs2_opt = hs2_opt.sort_values("HS2")
if hs2_opt.empty:
    st.warning("Không có HS2 nào có giá trị > 0."); st.stop()

c1, c2 = st.columns([2, 1])

# 
hs2_labels = hs2_opt["hs2_label"].tolist()
default_index = next((i for i, lab in enumerate(hs2_labels) if lab.startswith("85")), 0)

with c1:
    # chỉ tạo MỘT selectbox HS2 và dùng key mới
    hs2_pick = st.selectbox("Chọn mã HS2:", hs2_labels, index=default_index, key="hs2_pick_bottom")
    hs2_code = hs2_pick.split(" - ", 1)[0]

with c2:
    # dùng key khác cho detail để tránh trùng nếu ở nơi khác cũng có widget này
    detail_level = st.selectbox("Chọn cấp chi tiết:", ("HS4","HS6"), key="detail_level_bottom")

# ================== (C) HS4/HS6: TỔNG HỢP & VẼ SAU ==================
aggX, yrsX = agg_level(dfp, detail_level, hs2_code)

# cảnh báo nhẹ nếu một trong 2 năm không có dữ liệu ở cấp chi tiết
missing = [y for y in (y1,y2) if y not in yrsX]
if missing:
    st.info(f"{detail_level} trong HS2 {hs2_pick} không có dữ liệu cho năm: {', '.join(map(str, missing))}.")

p1_x,o1_x,t1_x = donut_parts(aggX, y1, topN)
p2_x,o2_x,t2_x = donut_parts(aggX, y2, topN)  
cmapX = cmap_from(o1_x,o2_x)

st.markdown(f"###### Phân tích {detail_level} (thuộc HS2 {hs2_pick})")
c,d = st.columns(2)
with c: st.caption(f"Cơ cấu {detail_level} – {y1} (Top {topN})"); plot_donut(p1_x,o1_x,y1,t1_x,cmapX,hide_legend)
with d: st.caption(f"Cơ cấu {detail_level} – {y2} (Top {topN})"); plot_donut(p2_x,o2_x,y2,t2_x,cmapX,hide_legend)



#----------------------------------------------------------------

st.subheader("1.3. So sánh tỷ trọng xuất nhập khẩu")

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
hs_level = st.selectbox("HS2/HS4/HS6:", available_levels)

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
    "Turkey nhập khẩu từ Việt Nam",
    "Tổng kim ngạch xuất khẩu của Việt Nam",
    "Tổng kim ngạch nhập khẩu của Turkey",
]

df_plot = (
    df_filtered[df_filtered["Criteria"].isin(criteria_list)]
    .groupby(["Year", "Criteria"], as_index=False)["Value (000 USD)"]
    .sum()
    .sort_values(["Criteria", "Year"])
)

# --- Bước 3: Vẽ biểu đồ Plotly (line) ---
st.markdown(f"###### So sánh giá trị thương mại {hs_selected_display} (2020–2024)")
fig = px.line(
    df_plot,
    x="Year",
    y="Value (000 USD)",
    color="Criteria",
    markers=True,
    #title=title_txt
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
col_tr_vn = "Turkey nhập khẩu từ Việt Nam"
col_vn_world = "Tổng kim ngạch xuất khẩu của Việt Nam"
col_tr_world = "Tổng kim ngạch nhập khẩu của Turkey"

# Tránh chia cho 0
denom1 = df_pivot[col_tr_world].replace(0, np.nan)
denom2 = df_pivot[col_vn_world].replace(0, np.nan)

ratio1 = (df_pivot[col_tr_vn] / denom1) * 100
ratio2 = (df_pivot[col_tr_vn] / denom2) * 100

# Làm tròn 2 chữ số & format %
ratio1_fmt = ratio1.round(2).map(lambda x: f"{x:,.3f}%" if pd.notna(x) else "")
ratio2_fmt = ratio2.round(2).map(lambda x: f"{x:,.3f}%" if pd.notna(x) else "")

tbl = pd.DataFrame({
    "Year": df_pivot.index,
    "Tỷ trọng (VN→TR / TR nhập từ thế giới)": ratio1_fmt.values,
    "Tỷ trọng (VN→TR / VN xuất ra thế giới)": ratio2_fmt.values,
}).set_index("Year").T

st.markdown(f"###### Bảng tỷ trọng (%) – 2020–2024")
sty = (tbl.style
       .set_properties(**{"border": "1px solid #e5e7eb"})
       .set_table_styles([{"selector": "th", "props": [("border","1px solid #e5e7eb")]}])
)
st.table(sty)

# (Không còn cần các cột tạm)
df.drop(columns=[c for c in ["_hs_code_str", "_hs_name_str"] if c in df.columns], inplace=True, errors="ignore")


st.header("Phần 2. Dữ liệu vận đơn năm 2024 (Nguồn: Tradesparq)")
st.write("*Lưu ý: Dữ liệu của trang Tradesparq có thể không đầy đủ*")

excel_path = r'D:\Dữ liệu XNK VN - TURKEY\Shipments_VN-TR.xlsx'

# Đường dẫn: nên dùng raw-string để tránh lỗi ký tự escape trên Windows
url = "https://raw.githubusercontent.com/thuthuy119/VNM-TUR_trade/main/Shipments_Jan-Apr.xlsx"
content = requests.get(url).content
df_bol = pd.read_excel(io.BytesIO(content), engine="openpyxl")

st.set_page_config(page_title="Phân tích lô hàng theo HS", layout="wide")

# ================== TIỀN XỬ LÝ ==================
@st.cache_data(show_spinner=False)
def prep_bol(df_bol: pd.DataFrame) -> pd.DataFrame:
    d = df_bol.copy()
    d.columns = d.columns.str.strip()

    # HS code → chỉ giữ số, tối đa 8 ký tự, pad trái; sinh HS2/4/6/8
    d["Hs Code"] = d["Hs Code"].astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)
    d["HS2"] = d["Hs Code"].str[:2]
    d["HS4"] = d["Hs Code"].str[:4]
    d["HS6"] = d["Hs Code"].str[:6]
    d["HS8"] = d["Hs Code"].str[:8]

    # Thời gian & numeric
    d["Date"] = pd.to_datetime(d.get("Date"), errors="coerce")
    for c in ["Amount($)", "Weight(Kg)", "Quantity", "TEU", "Freight fee", "Insurance fee"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)

    # Chuẩn hoá cột cảng: rỗng/NaN -> "Không có dữ liệu"
    for c in ["Origin Port", "Loading Place", "Destination Port", "Unloading Place"]:
        if c in d.columns:
            d[c] = (
                d[c].astype(str).str.strip()
                  .replace({"": np.nan})
                  .fillna("Không có dữ liệu")
            )
    return d

dfp = prep_bol(df_bol)

def first_existing(cols):  # lấy cột đầu tiên tồn tại
    return next((c for c in cols if c in dfp.columns), None)

COL_ORI  = first_existing(["Origin Port", "Loading Place"])
COL_DST  = first_existing(["Destination Port", "Unloading Place"])
COL_MODE = first_existing(["Mode of Transportation"])
COL_TERM = first_existing(["Trade Terms"])
DESC_COL = "HS Code Description" if "HS Code Description" in dfp.columns else None

# ================== 1) UI & BỘ CHỌN ==================
st.title("Phân tích lô hàng theo HS")



# Khoảng ngày (nếu có)
dmin, dmax = dfp["Date"].min(), dfp["Date"].max()
if pd.notna(dmin) and pd.notna(dmax):
    d1, d2 = st.date_input("Khoảng ngày", value=(dmin.date(), dmax.date()))
    dfp = dfp[(dfp["Date"].dt.date >= d1) & (dfp["Date"].dt.date <= d2)]

# Cấp HS (mặc định HS2) & chọn mã HS
choices = ["HS2", "HS4", "HS6", "HS8"]
hs_level = st.selectbox("Cấp HS", choices, index=choices.index("HS2"), key="hs_level")

opt = dfp[[hs_level]].dropna().drop_duplicates().sort_values(hs_level)
if DESC_COL:
    top_desc = dfp.groupby(hs_level)[DESC_COL].agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else "")
    opt = opt.merge(top_desc.rename("desc"), left_on=hs_level, right_index=True, how="left")
    opt["label"] = np.where(opt["desc"].fillna("")!="", opt[hs_level]+" - "+opt["desc"].astype(str), opt[hs_level])
else:
    opt["label"] = opt[hs_level]

labels = opt["label"].tolist() if not opt.empty else []
default_idx = next((i for i,l in enumerate(labels) if l.split(" - ",1)[0].startswith("85")), 0) if hs_level=="HS2" else 0

code_label = st.selectbox("Mã HS", labels, index=default_idx if labels else 0, key=f"hs_code_{hs_level}")
hs_code = code_label.split(" - ", 1)[0] if code_label else None


# ================== 2) LỌC THEO HS & TẠO SHIP_ID ==================
if not hs_code:
    st.info("Hãy chọn mã HS để xem biểu đồ.")
    st.stop()

# ⚙️ Tuỳ chọn hiển thị trong sidebar
legend_on = st.checkbox("Ẩn legend", value=False)

sub = dfp[dfp[hs_level] == hs_code].copy()

# Tạo SHIP_ID để đếm shipment chuẩn (nếu có đủ cột)
id_cols = [c for c in ["House Bill Number", "Master Bill Number",
                       "Customs Declaration Number", "Item Number"] if c in sub.columns]
sub["SHIP_ID"] = (sub[id_cols].astype(str).agg("|".join, axis=1)
                  if id_cols else sub.index.astype(str))

# ================== 3) HELPERS CHO DONUT ==================
def _clean_bycol(df: pd.DataFrame, by_col: str):
    if not by_col or by_col not in df.columns:
        return None
    ser = df[by_col].astype(str).str.strip().replace({"": np.nan}).dropna()
    out = df.loc[ser.index].copy()
    out[by_col] = ser
    return out

def donut_data_count_and_amount(df: pd.DataFrame, by_col: str):
    """Trả về (pie_count, order_count, total_count), (pie_amt, order_amt, total_amt)."""
    df2 = _clean_bycol(df, by_col)
    if df2 is None or df2.empty:
        return (pd.DataFrame(), [], 0.0), (pd.DataFrame(), [], 0.0)

    # Đếm số đơn hàng (SHIP_ID duy nhất) theo by_col
    pie_cnt = (df2.drop_duplicates([by_col, "SHIP_ID"])
                  .groupby(by_col, as_index=False)
                  .size().rename(columns={"size": "metric"})
                  .sort_values("metric", ascending=False))
    tot_cnt = float(pie_cnt["metric"].sum())

    # Tổng giá trị (Amount $)
    if "Amount($)" in df2.columns:
        pie_amt = (df2.groupby(by_col, as_index=False)["Amount($)"]
                      .sum().rename(columns={"Amount($)": "metric"})
                      .sort_values("metric", ascending=False))
        tot_amt = float(pie_amt["metric"].sum())
    else:
        pie_amt, tot_amt = pd.DataFrame(), 0.0

    if tot_cnt <= 0: pie_cnt = pd.DataFrame()
    if tot_amt <= 0: pie_amt = pd.DataFrame()
    order_cnt = pie_cnt[by_col].tolist() if not pie_cnt.empty else []
    order_amt = pie_amt[by_col].tolist() if not pie_amt.empty else []
    return (pie_cnt, order_cnt, tot_cnt), (pie_amt, order_amt, tot_amt)

def make_donut(pie_df: pd.DataFrame, order: list, title: str, total: float, value_label: str, show_legend: bool):
    if pie_df.empty:
        st.info(f"Không có dữ liệu cho {title}."); return
    name_col = pie_df.columns[0]
    cmap = {lab: px.colors.qualitative.Set2[i % 8] for i, lab in enumerate(order)}
    fig = px.pie(
        pie_df, names=name_col, values="metric", hole=0.62,
        category_orders={name_col: order},
        color=name_col, color_discrete_map=cmap,
        labels={name_col: "", "metric": value_label}
    )
    fig.update_traces(
        sort=False, textposition="inside", texttemplate="%{percent:.2%}",
        hovertemplate="<b>%{label}</b><br>" + value_label + ": %{value:,.0f}<br>Tỷ trọng: %{percent:.2%}<extra></extra>"
    )
    fig.update_layout(
        title=title,
        showlegend=show_legend,
        legend=dict(orientation="h", y=-0.12) if show_legend else None,
        margin=dict(t=42, l=6, r=6, b=60 if show_legend else 30),
        height=430,
        template="plotly_white",
        annotations=[dict(text=f"{total:,.0f}", x=0.5, y=0.5, showarrow=False, font=dict(size=17))]
    )
    st.plotly_chart(fig, use_container_width=True)

def render_donut_pair(df: pd.DataFrame, by_col: str, group_title_vi: str, show_legend: bool):
    st.subheader(group_title_vi)
    (pie_cnt, order_cnt, tot_cnt), (pie_amt, order_amt, tot_amt) = donut_data_count_and_amount(df, by_col)
    c1, c2 = st.columns(2)
    with c1:
        make_donut(
            pie_cnt, order_cnt,
            f"{group_title_vi} — theo số lượng đơn hàng",
            tot_cnt, value_label="Số đơn hàng",
            show_legend=show_legend
        )
    with c2:
        make_donut(
            pie_amt, order_amt,
            f"{group_title_vi} — theo giá trị",
            tot_amt, value_label="Giá trị (USD)",
            show_legend=show_legend
        )

# ================== 4) VẼ 4 NHÓM, MỖI NHÓM 2 DONUT ==================
render_donut_pair(sub, COL_ORI,  "Cảng đi",            legend_on)
render_donut_pair(sub, COL_DST,  "Cảng đến",           legend_on)
render_donut_pair(sub, COL_MODE, "Phương thức vận tải", legend_on)
render_donut_pair(sub, COL_TERM, "Điều khoản thương mại", legend_on)

# ================== 5) BẢNG TOP 20 DN ==================
EXPORTER_NAME = "Exporter"
IMPORTER_NAME = "Importer"

def _top20_table(df: pd.DataFrame, name_col: str, title_entity_vi: str):
    # title_entity_vi: "Nhà xuất khẩu" | "Nhà nhập khẩu"
    if not {name_col, "Amount($)"}.issubset(df.columns):
        st.info(f"Thiếu cột cần thiết để tạo bảng {title_entity_vi}."); 
        return

    df2 = _clean_bycol(df, name_col)
    if df2 is None or df2.empty:
        st.info(f"Không có dữ liệu cho {title_entity_vi}."); 
        return

    # Tổng số DN sau khi lọc theo HS & ngày
    total_firms = df2[name_col].nunique()

    # Tổng giá trị theo DN
    g = (df2.groupby(name_col, as_index=False)["Amount($)"]
             .sum().rename(columns={"Amount($)": "Giá trị"})
             .sort_values("Giá trị", ascending=False))

    total_market = float(g["Giá trị"].sum())
    top = g.head(20).copy()
    top["Thị phần (%)"] = (top["Giá trị"] / total_market * 100)

    # Chuẩn bị bảng hiển thị
    col_name_vi = "Nhà xuất khẩu" if name_col == "Exporter" else "Nhà nhập khẩu"
    display_df = (top[[name_col, "Giá trị", "Thị phần (%)"]]
                  .rename(columns={name_col: col_name_vi}))
    display_df["Giá trị"] = display_df["Giá trị"].round(0).map(lambda x: f"{x:,.0f}")
    display_df["Thị phần (%)"] = display_df["Thị phần (%)"].round(2)

    st.subheader(f"Top 20 {title_entity_vi}")
    st.markdown(f"**Tổng số {title_entity_vi}**: {total_firms:,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Hai bảng:
#_ top20_table = _top20_table  # giữ nguyên tên hàm gốc nếu cần dùng nơi khác
_top20_table(sub, EXPORTER_NAME, "Nhà xuất khẩu")
_top20_table(sub, IMPORTER_NAME, "Nhà nhập khẩu")

