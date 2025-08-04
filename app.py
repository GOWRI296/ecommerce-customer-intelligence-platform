import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🛍️ Shopper Scope ", layout="wide")
st.markdown('<h1 style="text-align: center; color: #FF6B6B;">🛍️ Shopper Scope </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 20px;">AI-Powered Product Recommendations & Customer Intelligence</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return (joblib.load('customer_model.pkl'), joblib.load('scaler.pkl'), 
            pd.read_csv('customers_with_groups.csv', index_col=0),
            np.load('product_similarities.npy'),
            pd.read_csv('customer_products.csv', index_col=0),
            pd.read_csv('product_names.csv', index_col=0))

model, scaler, customers, similarities, customer_products, product_names = load_data()

tab1, tab2 = st.tabs([" Product Recommendations", "👤 Customer Segment"])

with tab1:
    st.markdown("### Find Products Your Customers Will Love")
    
    products = list(customer_products.columns)
    selected = st.selectbox("🔍 Choose a product:", products)
    
    if st.button(" Get Recommendations", type="primary"):
        idx = products.index(selected)
        scores = similarities[idx]
        top5 = scores.argsort()[::-1][1:6]
        
        st.markdown("#### 🏆 Top 5 Similar Products:")
        
        for i, product_idx in enumerate(top5):
            product_code = products[product_idx]
            name = product_names.loc[product_code].values[0] if product_code in product_names.index else "Unknown"
            score = scores[product_idx]
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #4CAF50, #45a049); 
                        color: white; padding: 15px; margin: 10px 0; 
                        border-radius: 10px; border-left: 5px solid #2E7D32;">
                <h4>{i+1} 🎁 {product_code}</h4>
                <p>📝 {name[:40]}...</p>
                <p>⭐ Similarity Score: {score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Predict Customer Behavior")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.slider("📅 Days Since Last Purchase", 0, 365, 30)
    with col2:
        frequency = st.slider("🔄 Number of Orders", 1, 50, 5)
    with col3:
        monetary = st.slider("💰 Total Spent ($)", 0, 5000, 200)
    
    if st.button("🔮 Predict Customer Type", type="primary"):
        data = scaler.transform([[recency, frequency, monetary]])
        cluster = model.predict(data)[0]
        
        segments = {
            0: ("🔵 Regular Customer", "Steady and reliable shoppers", "#2196F3"),
            1: ("💎 VIP Customer", "Your most valuable customers!", "#4CAF50"), 
            2: ("⚠️ At-Risk Customer", "Need immediate attention", "#FF5722"),
            3: ("🌟 New Customer", "Fresh potential to nurture", "#FF9800")
        }
        
        title, desc, color = segments[cluster]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}, {color}AA); 
                    color: white; padding: 30px; text-align: center; 
                    border-radius: 20px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <h2>{title}</h2>
            <h4>{desc}</h4>
            <p>Cluster ID: {cluster}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if cluster == 1:
            st.success(" This customer is gold! Offer premium products and exclusive deals.")
        elif cluster == 2:
            st.error(" Send them a special discount to win them back!")
        elif cluster == 3:
            st.info(" Perfect time for welcome offers and onboarding.")
        else:
            st.info(" Keep them engaged with regular promotions.")

st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">Powered by AI & Machine Learning  (Gowri Nandhan) </p>', unsafe_allow_html=True)