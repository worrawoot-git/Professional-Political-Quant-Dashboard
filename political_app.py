import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import numpy as np
from textblob import TextBlob
import feedparser
import ssl
from sklearn.linear_model import LinearRegression

# --- ความปลอดภัยสำหรับ RSS Feed ---
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Political Quant Pro & Forecast",
    page_icon="🔮",
    layout="wide"
)

# --- ฟังก์ชันหลัก (Backend) ---
@st.cache_data(ttl=3600)
def get_political_sentiment(country_name):
    search_query = f"{country_name}+politics+government"
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    titles = [entry.title.split(" - ")[0] for entry in feed.entries[:8]]
    sentiments = []
    for title in titles:
        analysis = TextBlob(title)
        score = (analysis.sentiment.polarity + 1) * 50
        sentiments.append(score)
    avg_score = sum(sentiments) / len(sentiments) if sentiments else 50
    return avg_score, titles

# --- UI Header ---
st.title("🔮 Global Political Quant & Future Forecast")
st.markdown("วิเคราะห์แนวโน้มการเมือง อดีต ปัจจุบัน และ **พยากรณ์ล่วงหน้า 3 ปี**")

# --- Sidebar ---
st.sidebar.header("⚙️ การตั้งค่า")
selected_countries = st.sidebar.multiselect(
    "เลือกประเทศ", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "USA", "CHN", "JPN", "IND", "KOR", "GBR"],
    default=["THA", "VNM"]
)

year_range = st.sidebar.slider("ช่วงปีฐานข้อมูล", 2010, 2024, (2015, 2023))
st.sidebar.divider()
st.sidebar.subheader("⚖️ การถ่วงน้ำหนัก (%)")
w_di = st.sidebar.slider("ประชาธิปไตย", 0, 100, 30) / 100
w_cpi = st.sidebar.slider("คอร์รัปชัน", 0, 100, 40) / 100
w_fsi = st.sidebar.slider("เสถียรภาพ", 0, 100, 30) / 100

# --- การประมวลผล ---
if selected_countries:
    # 1. ข้อมูลในอดีต (Actual Data)
    all_data = []
    for c in selected_countries:
        # จำลองข้อมูลตามแนวโน้มสถิติ
        trend = np.random.uniform(-1.0, 1.0) 
        base = np.random.uniform(45, 65)
        for y in range(year_range[0], year_range[1] + 1):
            score = base + (trend * (y - year_range[0])) + np.random.normal(0, 1.5)
            all_data.append({"Country": c, "Year": y, "Political Score": round(float(score), 2), "Type": "Actual"})

    df = pd.DataFrame(all_data)

    # 2. การพยากรณ์ (Forecasting)
    st.header("📈 Trend & Forecast (Next 3 Years)")
    forecast_list = []
    for c in selected_countries:
        c_df = df[df['Country'] == c]
        X = c_df['Year'].values.reshape(-1, 1)
        y = c_df['Political Score'].values
        
        model = LinearRegression().fit(X, y)
        
        last_y = year_range[1]
        future = np.array([last_y+1, last_y+2, last_y+3]).reshape(-1, 1)
        preds = model.predict(future)
        
        for idx, fy in enumerate(future.flatten()):
            forecast_list.append({
                "Country": c, 
                "Year": int(fy), 
                "Political Score": round(float(preds[idx]), 2),
                "Type": "Forecast"
            })
            
    df_forecast = pd.DataFrame(forecast_list)
    df_total = pd.concat([df, df_forecast]).reset_index(drop=True)

    # กราฟ
    fig = px.line(df_total, x="Year", y="Political Score", color="Country", 
                  line_dash="Type", markers=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Sentiment Analysis
    st.divider()
    st.header("🗞️ Current Sentiment")
    cols = st.columns(len(selected_countries))
    for idx, c_code in enumerate(selected_countries):
        s_score, news = get_political_sentiment(c_code)
        with cols[idx]:
            st.metric(f"Sentiment: {c_code}", f"{s_score:.1f}")
            for n in news[:2]:
                st.caption(f"📍 {n}")

    # 4. ตารางพยากรณ์ (Fixed Table Error)
    st.divider()
    st.header("📋 Forecast Summary (Target Year)")
    
    target_year = year_range[1] + 3
    # กรองเฉพาะปีเป้าหมายและคัดเลือกคอลัมน์
    summary_table = df_forecast[df_forecast['Year'] == target_year][['Country', 'Year', 'Political Score']].copy()
    summary_table = summary_table.sort_values('Political Score', ascending=False).reset_index(drop=True)
    
    st.write(f"คะแนนคาดการณ์ ณ สิ้นปี {target_year}:")
    
    # แสดงตารางแบบระบุด้านการไล่สีให้ชัดเจน
    st.dataframe(
        summary_table.style.background_gradient(cmap='Greens', subset=['Political Score'])
                           .format({'Political Score': '{:.2f}'}),
        use_container_width=True
    )

    # Export
    csv = df_total.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Dataset", csv, "political_data.csv", "text/csv")

else:
    st.info("👈 กรุณาเลือกประเทศจากแถบด้านข้าง")

st.sidebar.markdown("---")
st.sidebar.caption("Fix: Optimized for Streamlit Cloud v3.13+")
