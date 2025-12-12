import time
import streamlit as st

import pandas as pd
from pandas import json_normalize
import requests


from io import BytesIO
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import datetime

from scipy import stats

import re

import itertools

url = st.secrets["API_URL"]

response = requests.get(url)
data = response.json()

st.set_page_config(page_title="類似患者検索", layout="wide")

st.title("類似患者検索")

def add_pre_levels(df):
  df['治療前PSRレベル'] = ''
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']>=90, 'レベル1')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<90, 'レベル2')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<85, 'レベル3')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<80, 'レベル4')

  df['治療前ASRレベル'] = ''
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']>=90, 'レベル1')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<90, 'レベル2')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<85, 'レベル3')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<80, 'レベル4')

  df['治療前CA重症度'] = '正常'
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>6, '軽症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>9, '中等症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>13, '重症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>17, '最重症')

  df['治療前CVAI重症度'] = '正常'
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>5, '軽症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>7, '中等症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>10, '重症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>14, '最重症')

  df['治療前短頭症'] = ''
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']>126, '長頭')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<=126, '正常')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<106, '軽症')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<103, '中等症')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<100, '重症')

  return(df)

def add_post_levels(df):
  df['最終PSRレベル'] = ''
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']>=90, 'レベル1')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<90, 'レベル2')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<85, 'レベル3')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<80, 'レベル4')

  df['最終ASRレベル'] = ''
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']>=90, 'レベル1')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<90, 'レベル2')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<85, 'レベル3')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<80, 'レベル4')

  df['最終CA重症度'] = '正常'
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>6, '軽症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>9, '中等症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>13, '重症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>17, '最重症')

  df['最終CVAI重症度'] = '正常'
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>5, '軽症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>7, '中等症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>10, '重症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>14, '最重症')

  df['最終短頭症'] = ''
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']>126, '長頭')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<=126, '正常')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<106, '軽症')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<103, '中等症')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<100, '重症')

  return(df)

#患者の月齢に応じて探索対象を制限する関数　2023/6/3 作成
def age_restriction(df, dfpt, tx=False):
  df_first = df.copy()
  mo = dfpt['月齢'].iloc[0]

  if tx:
    dftx = df_first
    dftx = dftx[dftx['治療ステータス'] == '治療前']

    if mo < 3:
      members = dftx[dftx['月齢']<mo+2]['ダミーID']
      #print('月齢を', mo+2, '未満に制限します')
    elif mo > 10:
      members=dftx[dftx['月齢']>mo-2]['ダミーID']
    else:
      members=dftx[(dftx['月齢']>mo-1)&(dftx['月齢']<mo+1)]['ダミーID']

    df_first = df_first[df_first['ダミーID'].isin(members)]

  else:
    if mo < 3:
      df_first=df_first[df_first['月齢']<mo+2]
    elif mo > 10:
      df_first=df_first[df_first['月齢']>mo-2]
    else:
      df_first=df_first[(df_first['月齢']>mo-1)&(df_first['月齢']<mo+1)]

  return(df_first)

def parameter_restriction(df, dfpt, tx=False):
  bi = dfpt['短頭率'].iloc[0]
  asr = dfpt['前頭部対称率'].iloc[0]
  psr = dfpt['後頭部対称率'].iloc[0]
  ca = dfpt['CA'].iloc[0]
  cvai =  dfpt['CVAI'].iloc[0]

  # if tx:
  #   df = df[(df['治療前短頭率'] > bi-3)&(df['治療前短頭率'] < bi+3)]
  #   df = df[(df['治療前前頭部対称率'] > asr-5)&(df['治療前前頭部対称率'] < asr+5)]
  #   df = df[(df['治療前後頭部対称率'] > psr-5)&(df['治療前後頭部対称率'] < psr+5)]
  #   df = df[(df['治療前CA'] > ca-4)&(df['治療前CA'] < ca+4)]
  #   df = df[(df['治療前CVAI'] > cvai-4)&(df['治療前CVAI'] < cvai+4)]

  # else:
  df = df[(df['短頭率'] > bi-3)&(df['短頭率'] < bi+3)]
  df = df[(df['前頭部対称率'] > asr-5)&(df['前頭部対称率'] < asr+5)]
  df = df[(df['後頭部対称率'] > psr-5)&(df['後頭部対称率'] < psr+5)]
  df = df[(df['CA'] > ca-4)&(df['CA'] < ca+4)]
  df = df[(df['CVAI'] > cvai-4)&(df['CVAI'] < cvai+4)]

  return(df)

def calc_visits(df, members):
  df_temp = df[df['ダミーID'].isin(members)]
  parameters = ['月齢','前後径','左右径','頭囲','短頭率','前頭部対称率','後頭部対称率','CA','CVAI']
  df_temp = df_temp.drop_duplicates(parameters, keep='last')

  # ダミーIDごとの件数 N を取得
  df_temp['通院回数'] = (
      df_temp
      .groupby('ダミーID')['治療期間']
      .transform('count') - 1
  )

  df_temp = df_temp.sort_values(['ダミーID', '治療期間'])
  df_temp = df_temp.drop_duplicates('ダミーID', keep='last')
  # return df_temp['治療期間'].describe()
  # return df_temp['通院回数'].describe()
  return df_temp[['ダミーID', '治療期間', '通院回数']]

# -------------------------
# 1) データ取得＆前処理をキャッシュ
# -------------------------
@st.cache_data(ttl=60*60, show_spinner=False)  # 1時間キャッシュ（必要に応じて調整）
def load_and_prepare_data(api_url: str):
    import numpy as np

    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["経過"])
    df_h = pd.DataFrame(data["ヘルメット"])

    # 数値化
    parameters_numeric = ['月齢', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', 'CA', '後頭部対称率', 'CVAI', 'CI']
    df[parameters_numeric] = df[parameters_numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().sort_values("月齢")

    df_h = df_h[(df_h["ダミーID"] != "") & (df_h["ヘルメット"] != "")]

    # 初診（治療前）1行
    df_first = df[df["治療ステータス"] == "治療前"].drop_duplicates("ダミーID")

    # 治療患者抽出
    treated_patients = df_h["ダミーID"].unique()
    df_tx = df[df["ダミーID"].isin(treated_patients)]

    df_tx_pre_last = df_tx[df_tx["治療ステータス"] == "治療前"].drop_duplicates("ダミーID", keep="last")
    df_tx_pre_last["治療前月齢"] = df_tx_pre_last["月齢"]
    df_tx_pre_last = add_pre_levels(df_tx_pre_last)

    df_tx_post = df_tx[df_tx["治療ステータス"] == "治療後"].copy()
    df_tx_pre_age = df_tx_pre_last[["ダミーID", "月齢"]].rename(columns={"月齢": "治療前月齢"})
    df_tx_post = pd.merge(df_tx_post, df_tx_pre_age, on="ダミーID", how="left")
    df_tx_post["治療期間"] = df_tx_post["月齢"] - df_tx_post["治療前月齢"]

    df_tx_pre_last["治療期間"] = 0
    df_tx_post = pd.merge(
        df_tx_post,
        df_tx_pre_last[["ダミーID","治療前PSRレベル","治療前ASRレベル","治療前短頭症","治療前CA重症度","治療前CVAI重症度"]],
        on="ダミーID",
        how="left",
    )

    df_tx_pre_post = pd.concat([df_tx_pre_last, df_tx_post], axis=0)
    df_tx_pre_post = pd.merge(df_tx_pre_post, df_h, on="ダミーID", how="left")

    df_tx_post_last = df_tx_post.drop_duplicates("ダミーID", keep="last").copy()
    df_tx_post_last = add_post_levels(df_tx_post_last)

    df_tx_pre_post = pd.merge(
        df_tx_pre_post,
        df_tx_post_last[["ダミーID","最終PSRレベル","最終ASRレベル","最終短頭症","最終CA重症度","最終CVAI重症度"]],
        on="ダミーID",
        how="left",
    )

    # 経過観察（再診群）整形
    df_first2 = add_pre_levels(df_first.copy())
    df_pre_age = df_first2[["ダミーID","月齢","治療前PSRレベル","治療前ASRレベル","治療前短頭症","治療前CA重症度","治療前CVAI重症度"]].rename(columns={"月齢":"治療前月齢"})
    df_co = pd.merge(df, df_pre_age, on="ダミーID", how="left")
    df_co = df_co[df_co["治療ステータス"] == "治療前"]
    obs_patients = df_co[df_co["ダミーID"].duplicated()]["ダミーID"].unique()
    df_co = df_co[df_co["ダミーID"].isin(obs_patients)].copy()

    age_diff_df = df_co.groupby("ダミーID")["月齢"].agg(["max","min"]).reset_index()
    age_diff_df["治療期間"] = age_diff_df["max"] - age_diff_df["min"]
    df_co = pd.merge(df_co, age_diff_df[["ダミーID","治療期間"]], on="ダミーID", how="left")

    df_co["ヘルメット"] = "経過観察"
    df_co["治療ステータス"] = df_co.groupby("ダミーID")["月齢"].transform(lambda x: ["治療前"] + ["治療後"]*(len(x)-1))
    df_co["ダミーID"] = df_co["ダミーID"] + "C"

    df_tx_pre_post = pd.concat([df_tx_pre_post, df_co], axis=0)

    # z スコア用
    parameters = ['月齢','前後径','左右径','頭囲','短頭率','CI','前頭部対称率','後頭部対称率','CA','CVAI']
    dfco_pre = df_co.drop_duplicates("ダミーID", keep="first").copy()
    df_first_z = df_first2.copy()
    df_tx_pre_last_z = df_tx_pre_last.copy()

    for dfx in [df_first_z, df_tx_pre_last_z, dfco_pre]:
        dfx["APR"] = dfx["前頭部対称率"]/dfx["後頭部対称率"]
        for i in parameters + ["APR"]:
            dfx["z_"+i] = (dfx[i] - dfx[i].mean())/dfx[i].std()

    treated_set = set(df_h["ダミーID"].astype(str).unique())
  
    # 以降、既存関数が参照する「グローバル」を返しておく
    return data, df, df_h, df_first_z, df_tx_pre_post, df_co, dfco_pre, treated_set


with st.sidebar:
    st.header("設定")
    api_url = st.secrets.get("API_URL", "")
    if not api_url:
        st.error("st.secrets['API_URL'] が設定されていません。")
        st.stop()

    if st.button("データ再読み込み（キャッシュ破棄）"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました。再読み込みしてください。")

    if st.button("解析結果をクリア"):
        for k in ["tx_rate_summary", "similar_summary", "tx_plot_fig", "tx_members", "co_plot_fig"]:
            st.session_state.pop(k, None)
        st.success("解析結果をクリアしました")

with st.spinner("APIからデータ取得＆前処理中...（初回は時間がかかります）"):
    data, df_all, df_h, df_first, df_tx_pre_post, df_co, dfco_pre, treated_set = load_and_prepare_data(api_url)

# 既存関数が参照している前提のグローバルを上書き（Colab移植の都合）
globals()["df_first"] = df_first
globals()["df_tx_pre_post"] = df_tx_pre_post
globals()["df_co"] = df_co
globals()["dfco_pre"] = dfco_pre

st.caption(f"経過データ: {len(df_all):,} 行 / 初診(治療前)ユニーク: {df_first['ダミーID'].nunique():,} 人 / 治療+経過観察統合: {df_tx_pre_post['ダミーID'].nunique():,} 人")

# -------------------------
# 2) 入力UI → dfpt作成
# -------------------------
st.subheader("お子様の入力")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    bd = st.date_input("生年月日", value=pd.to_datetime("2025-04-19").date())
    base_date = st.date_input("基準日（通常は今日）", value=pd.to_datetime(pd.Timestamp.today().date()).date())
with c2:
    front_back = st.number_input("前後径", value=137.3, step=0.1, format="%.1f")
    left_right = st.number_input("左右径", value=136.7, step=0.1, format="%.1f")
    head_circ = st.number_input("頭囲", value=432.7, step=0.1, format="%.1f")
with c3:
    ci_short = st.number_input("短頭率", value=100.4, step=0.1, format="%.1f")
    asr = st.number_input("前頭部対称率", value=96.3, step=0.1, format="%.1f")
    psr = st.number_input("後頭部対称率", value=88.6, step=0.1, format="%.1f")
    ca = st.number_input("CA", value=7.9, step=0.1, format="%.1f")
    cvai = st.number_input("CVAI", value=5.5, step=0.1, format="%.1f")

# 月齢計算（Colabと同じ 30.4375日/月）
m = (pd.to_datetime(base_date) - pd.to_datetime(bd)) / pd.Timedelta(30.4375, "D")
m = float(m)

dfpt = pd.DataFrame(data={
    "月齢":[m],
    "前後径":[front_back],
    "左右径":[left_right],
    "頭囲":[head_circ],
    "短頭率":[ci_short],
    "前頭部対称率":[asr],
    "後頭部対称率":[psr],
    "CA":[ca],
    "CVAI":[cvai],
    # Colab互換のダミー列（必要なら）
    "頭蓋体積":[""],
    "耳介偏位":[""],
    "APR":[""],
    "CHI":[""],
    "AFI":[""],
    "PFI":[""],
    "TBI":[""],
    "後屈度":[""],
})

# st.write("入力データ（dfpt）")
# st.dataframe(dfpt, use_container_width=True)

def tx_rate_st(dfpt, df_first=df_first, treated_set=None, n=30):
    if treated_set is None:
        treated_set = set(df_h["ダミーID"].astype(str).unique())
      
    st.write("### 類似症例の治療率")

    dfpt = dfpt.copy()
    df_first = df_first.copy()

    dfpt["APR"] = dfpt["前頭部対称率"] / dfpt["後頭部対称率"]
    df_first["APR"] = df_first["前頭部対称率"] / df_first["後頭部対称率"]

    parameters = ["月齢","前後径","左右径","頭囲","短頭率","前頭部対称率","後頭部対称率","CA","CVAI","APR"]

    df_first_para = df_first[parameters].reset_index(drop=True)
    dfpt_z = (dfpt[parameters] - df_first_para.mean()) / df_first_para.std()
    dfpt_w = 10 ** abs(dfpt_z)

    # 月齢重みを最大に
    if dfpt_w["月齢"].iloc[0] < dfpt_w.T.max().iloc[0]:
        dfpt_w.loc[dfpt_w.index[0], "月齢"] = dfpt_w.T.max().iloc[0]

    df_first2 = age_restriction(df_first, dfpt)
    df_first_temp = parameter_restriction(df_first2, dfpt)  # ※この関数は別途定義が必要
    if len(df_first_temp) >= 10:
        df_first2 = df_first_temp

    st.write(f"探索対象人数: **{len(df_first2)} 人**")

    df_first2 = df_first2.copy()
    df_first2["w_delta"] = 0
    for p in parameters:
        df_first2["w_delta"] += dfpt_w[p].iloc[0] * abs(df_first2["z_"+p] - dfpt_z[p].iloc[0])**2

    # 最適N探索（N=1を避ける）
    min_N = 5  # ここを 3 や 10 にしてもOK（要件に合わせて）
    max_N = min(n, len(df_first2))  # 探索可能な上限
    
    # 候補が少ないときの保険
    if max_N < min_N:
        min_N = max_N  # 取れるだけ取る（0になるなら0扱い）
    
    d = 1e10
    N = min_N if min_N >= 1 else 0
    
    for i in range(min_N, max_N + 1):
        dfalln = df_first2.sort_values("w_delta").head(i)
        dfn = dfalln[parameters].agg(["mean"]).reset_index(drop=True)
        sum_delta = ((dfpt[parameters] - dfn)**2).sum(axis=1).iloc[0]
        if sum_delta < d:
            d = sum_delta
            N = i

    st.write(f"最適人数: **{N} 人**（誤差スコア {round(d,2)}）")

    dfallN = df_first2.sort_values("w_delta").head(N)
    
    # treated_patients（df_h由来の治療実施者ID）で数える
    # treated_set = set(df_h["ダミーID"].astype(str).unique())
    similar_ids = dfallN["ダミーID"].astype(str).unique()
    
    ntx = sum(pid in treated_set for pid in similar_ids)
    tx_rate = (ntx / len(similar_ids) * 100) if len(similar_ids) > 0 else 0.0

    # st.write(f"治療率: **{round(tx_rate,1)}%**（{ntx}人 / {N}人）")
    st.write(f"治療率: **{round(tx_rate,1)}%**（{ntx}人 / {len(similar_ids)}人）")

    visits = _visits_summary(df_tx_pre_post, dfallN["ダミーID"].unique())
    return {
        "探索対象人数": int(len(df_first2)),
        "最適人数N": int(N),
        "治療率(%)": float(tx_rate),
        "治療人数": int(ntx),
        "誤差スコア": float(d),
        "治療期間_mean": visits["治療期間_mean"] if visits else None,
        "治療期間_std": visits["治療期間_std"] if visits else None,
        "通院回数_mean": visits["通院回数_mean"] if visits else None,
        "通院回数_std": visits["通院回数_std"] if visits else None,
    }


def similar_pts_st(dfpt, min=5, remove_self=False):
    # st.write("### similar_pts 実行ログ")

    dfpt = dfpt.copy()
    dfpt["APR"] = dfpt["前頭部対称率"] / dfpt["後頭部対称率"]

    parameters = ["月齢","前後径","左右径","頭囲","短頭率","前頭部対称率","後頭部対称率","CA","CVAI","APR"]

    dftx_pre = df_tx_pre_post[df_tx_pre_post["治療ステータス"] == "治療前"].copy()
    dftx_pre["APR"] = dftx_pre["前頭部対称率"] / dftx_pre["後頭部対称率"]

    if remove_self:
        dftx_pre = dftx_pre[~(dftx_pre[parameters] == dfpt[parameters].iloc[0]).all(axis=1)]

    dftx_pre2 = age_restriction(dftx_pre, dfpt[parameters], tx=True)
    dftx_pre_temp = parameter_restriction(dftx_pre2, dfpt[parameters], tx=True)  # ※別途定義が必要
    if len(dftx_pre_temp) >= 10:
        dftx_pre2 = dftx_pre_temp

    # st.write(f"探索対象人数: **{dftx_pre2['ダミーID'].nunique()} 人**")

    dfpt_z = (dfpt[parameters] - dftx_pre2[parameters].mean()) / dftx_pre2[parameters].std()
    dfpt_w = 10 ** abs(dfpt_z)

    if dfpt_w["月齢"].iloc[0] < dfpt_w.T.max().iloc[0]:
        dfpt_w["月齢"] = dfpt_w.T.max().iloc[0]

    dftx_pre2 = dftx_pre2.copy()
    dftx_pre2["w_delta"] = 0
    for p in parameters:
        dftx_pre2["w_delta"] += dfpt_w[p].iloc[0] * abs(df_first["z_"+p] - dfpt_z[p].iloc[0])**2

    d = 1e10
    N = min
    for i in range(min, 30):
        dfpren = dftx_pre2.sort_values("w_delta").head(i)
        dfn = dfpren[parameters].agg(["mean"]).reset_index(drop=True)
        sum_delta = ((dfpt[parameters] - dfn)**2).sum(axis=1).iloc[0]
        if sum_delta < d:
            d = sum_delta
            N = i

    # st.write(f"最適人数: **{N} 人**")

    members = dftx_pre2.sort_values("w_delta").head(N)["ダミーID"].unique()
    visits = _visits_summary(df_tx_pre_post, members)

    return {
        "探索対象人数": int(dftx_pre2["ダミーID"].nunique()),
        "最適人数N": int(N),
        "誤差スコア": float(d),
        "治療期間_mean": visits["治療期間_mean"] if visits else None,
        "治療期間_std": visits["治療期間_std"] if visits else None,
        "通院回数_mean": visits["通院回数_mean"] if visits else None,
        "通院回数_std": visits["通院回数_std"] if visits else None,
        "members": list(members),
    }


def co_plot_fig(dfpt):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # hoverで見たい項目
    HOVER_COLS = ["月齢","頭囲","短頭率","前頭部対称率","後頭部対称率","CA","CVAI","前後径","左右径"]

    dfpt = dfpt.copy()
    dfpt["APR"] = dfpt["前頭部対称率"] / dfpt["後頭部対称率"]

    parameters = ["月齢","前後径","左右径","頭囲","短頭率",
                  "前頭部対称率","後頭部対称率","CA","CVAI","APR"]

    dfco_pre_global = dfco_pre.copy()

    dfpt_z = (dfpt[parameters] - dfco_pre_global[parameters].mean()) / dfco_pre_global[parameters].std()
    dfpt_w = 10 ** abs(dfpt_z)

    if dfpt_w["月齢"].iloc[0] < dfpt_w.T.max().iloc[0]:
        dfpt_w["月齢"] = dfpt_w.T.max().iloc[0]

    dfco_pre_global["w_delta"] = 0
    for p in parameters:
        dfco_pre_global["w_delta"] += (
            dfpt_w[p].iloc[0] * abs(df_first["z_" + p] - dfpt_z[p].iloc[0]) ** 2
        )

    rank = list(dfco_pre_global.sort_values("w_delta")["ダミーID"])[:10]
    dfcon = df_co[df_co["ダミーID"].isin(rank)]

    para_table = [
        ["頭囲", "短頭率"],
        ["前頭部対称率", "後頭部対称率"],
        ["CA", "CVAI"],
        ["前後径", "左右径"],
    ]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=sum(para_table, [])
    )

    # hover用テンプレート
    hover_cols = [c for c in HOVER_COLS if c in dfpt.columns]

    def hovertemplate(prefix: str):
        lines = [f"{prefix}<br>"]
        for i, c in enumerate(hover_cols):
            if c == "月齢":
                lines.append(f"{c}=%{{customdata[{i}]:.2f}}<br>")
            else:
                lines.append(f"{c}=%{{customdata[{i}]:.1f}}<br>")
        lines.append("<extra></extra>")
        return "".join(lines)

    # ---------- お子様 ----------
    custom_pt = dfpt[hover_cols].to_numpy()
    for i in range(4):
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=dfpt["月齢"],
                    y=dfpt[para_table[i][j]],
                    mode="lines+markers",
                    marker=dict(color="green", size=9),
                    line=dict(width=3),
                    showlegend=False,
                    customdata=custom_pt,
                    hovertemplate=hovertemplate("お子様"),
                ),
                row=i+1, col=j+1
            )

    # ---------- 経過観察患者 ----------
    colors = px.colors.qualitative.Alphabet
    c = 0
    for pid in dfcon["ダミーID"].unique():
        tmp = dfcon[dfcon["ダミーID"] == pid].copy()
        if tmp.empty:
            continue

        hover_cols_sim = [c for c in hover_cols if c in tmp.columns]
        custom_sim = tmp[hover_cols_sim].to_numpy()

        for i in range(4):
            for j in range(2):
                fig.add_trace(
                    go.Scatter(
                        x=tmp["月齢"],
                        y=tmp[para_table[i][j]],
                        mode="lines+markers",
                        marker=dict(size=5, color=colors[c % len(colors)]),
                        line=dict(width=1),
                        showlegend=False,
                        customdata=custom_sim,
                        hovertemplate=hovertemplate(f"ID={pid}"),
                    ),
                    row=i+1, col=j+1
                )
        c += 1

    fig.update_layout(
        height=1300,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def _visits_summary(df_tx_pre_post: pd.DataFrame, members):
    """members(ダミーID一覧) から 治療期間/通院回数 を患者単位で集計して mean/std を返す"""
    if len(members) == 0:
        return None

    tmp = df_tx_pre_post[df_tx_pre_post["ダミーID"].isin(members)].copy()
    if tmp.empty or "治療期間" not in tmp.columns:
        return None

    tmp["治療期間"] = pd.to_numeric(tmp["治療期間"], errors="coerce")

    g = (
        tmp.groupby("ダミーID", as_index=False)
           .agg(
               治療期間=("治療期間", "max"),
               通院回数=("治療期間", lambda x: x.dropna().nunique() - 1)
           )
    )

    # 表示用の通院回数は +1
    visits_count = g["通院回数"] + 1

    res = {
        "治療期間_mean": float(g["治療期間"].mean()),
        "治療期間_std": float(g["治療期間"].std(ddof=1)) if len(g) >= 2 else 0.0,
        "通院回数_mean": float(visits_count.mean()),
        "通院回数_std": float(visits_count.std(ddof=1)) if len(g) >= 2 else 0.0,
        "n_patients": int(g.shape[0]),
    }
    return res


# -------------------------
# 3) Streamlit 用の tx_plot（figを返す版）
#    ※ Colabの tx_plot のロジックを “表示部分だけ” Streamlit向けに変更
# -------------------------
def tx_plot_fig(dfpt_in: pd.DataFrame, dftx: pd.DataFrame, n=10, mo_weight=1):
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # hoverで見たい項目（存在しない列は自動で除外）
    HOVER_COLS = ["月齢","頭囲","短頭率","前頭部対称率","後頭部対称率","CA","CVAI","前後径","左右径"]

    dfpt = dfpt_in.copy()
    dftx = dftx.copy()

    # APR
    dfpt["APR"] = dfpt["前頭部対称率"] / dfpt["後頭部対称率"]
    dftx["APR"] = dftx["前頭部対称率"] / dftx["後頭部対称率"]

    dftx_pre = dftx[dftx["治療ステータス"] == "治療前"].copy()
    dftx_pre["APR"] = dftx_pre["前頭部対称率"] / dftx_pre["後頭部対称率"]

    parameters = ["月齢","前後径","左右径","頭囲","短頭率","前頭部対称率","後頭部対称率","CA","CVAI","APR"]
    dfpt_temp = dfpt[parameters]

    # z
    dftx_pre_para = dftx_pre[parameters]
    dfpt_z = (dfpt_temp - dftx_pre_para.mean()) / dftx_pre_para.std()
    dfpt_w = 10 ** abs(dfpt_z)

    # 月齢重み
    if dfpt_w["月齢"].iloc[0] < dfpt_w.T.max().iloc[0]:
        dfpt_w["月齢"] = dfpt_w.T.max().iloc[0]
    dfpt_w["月齢"] = dfpt_w["月齢"] * mo_weight

    # 絞り込み（関数は別途定義済み前提）
    dftx_pre2 = age_restriction(dftx_pre, dfpt_temp, tx=True)
    dftx_pre_temp = parameter_restriction(dftx_pre2, dfpt_temp, tx=True)
    if len(dftx_pre_temp) >= 10:
        dftx_pre2 = dftx_pre_temp

    # w_delta
    dftx_pre2 = dftx_pre2.copy()
    dftx_pre2["w_delta"] = 0
    for p in parameters:
        dftx_pre2["w_delta"] += dfpt_w[p].iloc[0] * abs(df_first["z_" + p] - dfpt_z[p].iloc[0]) ** 2

    rank = list(dftx_pre2.sort_values("w_delta")["ダミーID"])
    similar_patients = rank[:n]

    dftxn = dftx[dftx["ダミーID"].isin(similar_patients)].copy()

    # 画面表示は上位10人だけ（必要なら増やす）
    display_ids = list(dftxn["ダミーID"].unique()[:10])

    para_table = [
        ["頭囲", "短頭率"],
        ["前頭部対称率", "後頭部対称率"],
        ["CA", "CVAI"],
        ["前後径", "左右径"],
    ]

    fig = make_subplots(
        rows=len(para_table),
        cols=len(para_table[0]),
        subplot_titles=sum(para_table, []),
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    # hoverに使える列だけに絞る（ない列で落ちないように）
    hover_cols = [c for c in HOVER_COLS if c in dfpt.columns]

    def _hovertemplate(prefix: str):
        # customdata の順番は hover_cols と一致
        # 例: prefix="お子様" または f"ID={pid}"
        lines = [f"{prefix}<br>"]
        for k, col in enumerate(hover_cols):
            # 月齢だけ小数、他は1桁小数に寄せる（必要なら調整）
            if col == "月齢":
                lines.append(f"{col}=%{{customdata[{k}]:.2f}}<br>")
            else:
                lines.append(f"{col}=%{{customdata[{k}]:.1f}}<br>")
        lines.append("<extra></extra>")
        return "".join(lines)

    # x軸範囲
    if len(dftxn):
        xmin = float(dftxn["月齢"].min() - 0.1)
        xmax = float(dftxn["月齢"].max() + 0.1)
    else:
        xmin = float(dfpt["月齢"].iloc[0] - 0.1)
        xmax = float(dfpt["月齢"].iloc[0] + 0.1)

    # ------- お子様（legend非表示・hover強化） -------
    custom_pt = dfpt[hover_cols].to_numpy()
    for i in range(len(para_table)):
        for j in range(len(para_table[i])):
            ycol = para_table[i][j]
            fig.add_trace(
                go.Scatter(
                    x=dfpt["月齢"],
                    y=dfpt[ycol],
                    mode="lines+markers",
                    marker=dict(size=9, color="green"),
                    line=dict(width=3),
                    showlegend=False,
                    customdata=custom_pt,
                    hovertemplate=_hovertemplate("お子様"),
                ),
                row=i + 1, col=j + 1,
            )

    # ------- 類似患者（legend非表示・hover強化） -------
    list_colors = px.colors.qualitative.Alphabet
    c = 0
    for pid in display_ids:
        tmp = dftxn[dftxn["ダミーID"] == pid].copy()
        if tmp.empty:
            continue

        # hover列が無い場合に備える
        hover_cols_sim = [c for c in hover_cols if c in tmp.columns]
        custom_sim = tmp[hover_cols_sim].to_numpy()

        for i in range(len(para_table)):
            for j in range(len(para_table[i])):
                ycol = para_table[i][j]
                fig.add_trace(
                    go.Scatter(
                        x=tmp["月齢"],
                        y=tmp[ycol],
                        mode="lines+markers",
                        marker=dict(size=5, color=list_colors[c % len(list_colors)]),
                        line=dict(width=1),
                        showlegend=False,
                        customdata=custom_sim,
                        hovertemplate=_hovertemplate(f"ID={pid}"),
                    ),
                    row=i + 1, col=j + 1,
                )
        c += 1

    # ------- 見やすさ調整（legend消す、サイズ、余白） -------
    fig.update_layout(
        showlegend=False,                 # 完全非表示
        height=1300,                       # ここで全体の高さを抑える
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # 軸（共通）
    for i in range(len(para_table)):
        for j in range(len(para_table[i])):
            # fig.update_xaxes(title="月齢", range=[xmin, xmax], row=i + 1, col=j + 1)
          fig.update_xaxes(range=[xmin, xmax], row=i + 1, col=j + 1)

    return fig, similar_patients



# -------------------------
# 4) 実行ボタン
# -------------------------
st.subheader("実行")

run_all = st.button("解析を実行", use_container_width=True)

if run_all:
    with st.spinner("計算中...（初回は時間がかかります）"):
        # 実行して session_state に保存（再描画で再計算しない）
        st.session_state["tx_rate_summary"] = tx_rate_st(dfpt.copy(), df_first=df_first, n=30)
        st.session_state["similar_summary"] = similar_pts_st(dfpt.copy(), min=5, remove_self=False)


        n = 100
        mo_weight = 10
        fig_tx, members = tx_plot_fig(dfpt.copy(), df_tx_pre_post, n=n, mo_weight=mo_weight)
        st.session_state["tx_plot_fig"] = fig_tx
        st.session_state["tx_members"] = members

        fig_co = co_plot_fig(dfpt.copy())
        st.session_state["co_plot_fig"] = fig_co

# 表示（ボタン押さなくても session_state にあれば出す）
# if "tx_rate_summary" in st.session_state:
#     st.markdown("## tx_rate（平均・分散）")
#     s = st.session_state["tx_rate_summary"]
#     st.write(f"治療率: {s['治療率(%)']:.1f}%  / 最適N={s['最適人数N']}  / 探索対象={s['探索対象人数']}")
#     st.write(f"治療期間 平均={s['治療期間_mean']:.2f}  分散={s['治療期間_var']:.2f}")
#     st.write(f"通院回数 平均={s['通院回数_mean']:.2f}  分散={s['通院回数_var']:.2f}")

if "similar_summary" in st.session_state:
    st.markdown("## 治療患者の通院期間・回数")
    s = st.session_state["similar_summary"]
    st.write(f"最適N={s['最適人数N']} / 探索対象={s['探索対象人数']}")

    m = s.get("治療期間_mean")
    sd = s.get("治療期間_std")
    if m is not None and sd is not None:
      # st.write(f"通院期間（平均±標準偏差）= **{m:.1f} ± {sd:.1f} か月**")
      st.markdown(f"通院期間（平均±標準偏差）= **{m:.1f} ± {sd:.1f} か月**")

    m = s.get("通院回数_mean")
    sd = s.get("通院回数_std")
    if m is not None and sd is not None:
      # st.write(f"通院回数（平均±標準偏差）= **{m:.1f} ± {sd:.1f} 回**")
      st.markdown(f"通院回数（平均±標準偏差）= **{m:.1f} ± {sd:.1f} 回**")

if "tx_plot_fig" in st.session_state:
    st.markdown("## 治療患者の経過")
    st.plotly_chart(st.session_state["tx_plot_fig"], use_container_width=True)

if "co_plot_fig" in st.session_state:
    st.markdown("## 経過観察患者の経過")
    st.plotly_chart(st.session_state["co_plot_fig"], use_container_width=True)



# st.subheader("実行")

# run1, run2, run3, run4 = st.columns([1,1,1,1])

# with run1:
#     do_tx_rate = st.button("治療率推定（tx_rate）", use_container_width=True)
# with run2:
#     do_similar = st.button("治療患者から類似抽出（similar_pts）", use_container_width=True)
# with run3:
#     do_plot = st.button("類似経過プロット（tx_plot）", use_container_width=True)
# with run4:
#     do_co_plot = st.button("経過観察の類似プロット（co_plot）", use_container_width=True)

# # -------------------------
# # tx_rate_st（ログ表示あり）
# # -------------------------
# if do_tx_rate:
#     with st.spinner("tx_rate 実行中..."):
#         try:
#             with st.expander("tx_rate ログ", expanded=True):
#                 df_res = tx_rate_st(dfpt.copy(), df_first=df_first, n=30)
#             st.success("完了")
#             st.dataframe(df_res, use_container_width=True)
#         except Exception as e:
#             st.exception(e)

# # -------------------------
# # similar_pts_st（ログ表示あり）
# # -------------------------
# if do_similar:
#     with st.spinner("similar_pts 実行中..."):
#         try:
#             with st.expander("similar_pts ログ", expanded=True):
#                 df_sim = similar_pts_st(dfpt.copy(), min=5, remove_self=False)
#             st.success("完了")
#             st.dataframe(df_sim, use_container_width=True)
#         except Exception as e:
#             st.exception(e)

# # -------------------------
# # tx_plot_fig（既に fig 返す版でOK）
# # -------------------------
# if do_plot:
#     with st.spinner("tx_plot 実行中..."):
#         try:
#             # sliderはボタン外に出すと毎回再描画で値が変わって見えるので、ここに置くのが無難
#             n = st.slider("表示に使う類似患者数 n", min_value=10, max_value=500, value=100, step=10, key="txplot_n")
#             mo_weight = st.slider("月齢重み mo_weight", min_value=1, max_value=20, value=10, step=1, key="txplot_mw")

#             fig, members = tx_plot_fig(dfpt.copy(), df_tx_pre_post, n=n, mo_weight=mo_weight)

#             st.success(f"完了（抽出: {len(members)}人）")
#             st.plotly_chart(fig, use_container_width=True)
#             st.write("抽出されたID（先頭）", members[:20])
#         except Exception as e:
#             st.exception(e)

# # -------------------------
# # co_plot_fig（fig返す版 → Streamlitで表示）
# # -------------------------
# if do_co_plot:
#     with st.spinner("co_plot 実行中..."):
#         try:
#             fig = co_plot_fig(dfpt.copy())
#             st.success("完了")
#             st.plotly_chart(fig, use_container_width=True)
#         except Exception as e:
#             st.exception(e)

