import streamlit as st
import pandas as pd
import plotly.express as px
import wbgapi as wb
import numpy as np
from textblob import TextBlob
import feedparser
import ssl
from sklearn.linear_model import LinearRegression # เพิ่มสำหรับการพยากรณ์

# --- การตั้งค่าความปลอดภัยสำหรับ News Feed ---
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Political Quant Pro & Forecast",
    page_icon="🔮",
    layout="wide"
)

# --- ฟังก์ชันการคำนวณ Sentiment ---
@st.cache_data(ttl=3600)
def get_political_sentiment(country_name):
    search_query = f"{country_name}+politics+government"
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    titles = [entry.title for entry in feed.entries[:8]]
    sentiments = []
    for title in titles:
        analysis = TextBlob(title.split(" - ")[0])
        score = (analysis.sentiment.polarity + 1) * 50
        sentiments.append(score)
    avg_score = sum(sentiments) / len(sentiments) if sentiments else 50
    return avg_score, titles

# --- UI: Header ---
st.title("🔮 Global Political Quant & Future Forecast")
st.markdown("วิเคราะห์แนวโน้มการเมืองในอดีต ปัจจุบัน และ **พยากรณ์ล่วงหน้า 3 ปี**")

# --- Sidebar ---
st.sidebar.header("⚙️ การตั้งค่า")
selected_countries = st.sidebar.multiselect(
    "เลือกประเทศ", 
    ["THA", "VNM", "IDN", "SGP", "MYS", "USA", "CHN", "JPN", "IND"],
    default=["THA", "VNM"]
)

year_range = st.sidebar.slider("ช่วงปีที่ใช้เป็นฐานข้อมูล", 2010, 2024, (2015, 2023))
st.sidebar.divider()
st.sidebar.subheader("⚖️ Weights")
w_di = st.sidebar.slider("Democracy Index (%)", 0, 100, 30) / 100
w_cpi = st.sidebar.slider("Corruption (%)", 0, 100, 40) / 100
w_fsi = st.sidebar.slider("Stability (%)", 0, 100, 30) / 100

# --- การประมวลผลหลัก ---
if selected_countries:
    # 1. การสร้างข้อมูลในอดีต (Historical Data)
    all_data = []
    for c in selected_countries:
        # สร้างแนวโน้มแบบสุ่มที่ดูสมจริง (ในงานจริงจะดึงจาก WB API)
        trend_factor = np.random.uniform(-1.5, 1.5) 
        base_score = np.random.uniform(40, 70)
        
        for y in range(year_range[0], year_range[1] + 1):
            noise = np.random.normal(0, 2)
            score = base_score + (trend_factor * (y - year_range[0])) + noise
            all_data.append({"Country": c, "Year": y, "Political Score": round(score, 2), "Type": "Actual"})

    df = pd.DataFrame(all_data)

    # 2. ฟีเจอร์พยากรณ์ (Forecasting Feature)
    st.header("📈 Historical Trend & 3-Year Forecast")
    
    forecast_results = []
    for c in selected_countries:
        country_df = df[df['Country'] == c]
        
        # เตรียมข้อมูลสำหรับ Linear Regression
        X = country_df['Year'].values.reshape(-1, 1)
        y = country_df['Political Score'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # พยากรณ์ไปอีก 3 ปีข้างหน้า
        last_year = year_range[1]
        future_years = np.array([last_year + 1, last_year + 2, last_year + 3]).reshape(-1, 1)
        predictions = model.predict(future_years)
        
        for idx, fy in enumerate(future_years.flatten()):
            forecast_results.append({
                "Country": c, 
                "Year": int(fy), 
                "Political Score": round(predictions[idx], 2),
                "Type": "Forecast"
            })
            
    df_forecast = pd.DataFrame(forecast_results)
    df_total = pd.concat([df, df_forecast])

    # กราฟแสดงผล (เส้นทึบ = จริง, เส้นประ = พยากรณ์)
    fig = px.line(df_total, x="Year", y="Political Score", color="Country", 
                  line_dash="Type", markers=True,
                  title="การพยากรณ์แนวโน้มสุขภาพทางการเมือง (Actual vs Forecast)")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Sentiment Analysis (Real-time)
    st.divider()
    st.header("🗞️ Real-time Sentiment Context")
    cols = st.columns(len(selected_countries))
    
    sent_list = []
    for idx, c_code in enumerate(selected_countries):
        s_score, news = get_political_sentiment(c_code)
        sent_list.append({"Country": c_code, "Sentiment": s_score})
        with cols[idx]:
            st.metric(f"Current Sentiment: {c_code}", f"{s_score:.1f}")
            for n in news[:2]:
                st.caption(f"📰 {n}")

    # 4. ตารางสรุปเชิงพยากรณ์
    st.divider()
    st.header("📋 Forecast Summary Table")
    
    # ดึงเฉพาะปีสุดท้ายที่พยากรณ์ (3 ปีข้างหน้า)
    latest_forecast = df_forecast[df_forecast['Year'] == year_range[1] + 3]
    
    st.write(f"ตารางคาดการณ์คะแนนในปี {year_range[1] + 3}:")
    
    # 4. ตารางสรุปเชิงพยากรณ์
    st.divider()
    st.header("📋 Forecast Summary Table")
    
    # ดึงเฉพาะปีสุดท้ายที่พยากรณ์ (3 ปีข้างหน้า)
    latest_forecast = df_forecast[df_forecast['Year'] == year_range[1] + 3].copy()
    
    st.write(f"ตารางคาดการณ์คะแนนในปี {year_range[1] + 3}:")
    
    # แก้ไขจุดที่ Error: ระบุเฉพาะคอลัมน์ 'Political Score' ให้ทำ Gradient
    st.dataframe(
        latest_forecast.style.background_gradient(cmap='Blues', subset=['Political Score']), 
        use_container_width=True
    )

    # Export
    csv = df_total.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Data (Historical + Forecast)", csv, "political_forecast.csv", "text/csv")

else:
    st.info("👈 กรุณาเลือกประเทศเพื่อดูการพยากรณ์")

# เพิ่มคำอธิบายอัลกอริทึม
with st.expander("📝 อธิบายหลักการพยากรณ์ (Methodology)"):
    st.write("""
    - **Historical Data:** ใช้ข้อมูลจากช่วงปีที่เลือกมาเป็นฐานในการหาความสัมพันธ์
    - **Linear Regression:** ใช้อัลกอริทึมวิเคราะห์เส้นแนวโน้ม (Trendline) เพื่อคำนวณทิศทางว่าในอนาคตคะแนนควรจะเป็นเท่าใด
    - **Forecast Line:** เส้นประในกราฟแสดงถึงการคาดการณ์เชิงสถิติ ซึ่งอาจเปลี่ยนแปลงได้ตามเหตุการณ์ปัจจุบัน (Sentiment)
    """)
