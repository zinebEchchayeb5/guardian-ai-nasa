import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# -----------------------------------------------------------------------------
# 0. CONFIGURATION AND CONSTANTS
# -----------------------------------------------------------------------------

# Streamlit page configuration
st.set_page_config(
    page_title="Guardian AI: Advanced AQI Forecasts",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Constants (Replace with your keys for real usage)
OPENAQ_API_URL = "https://api.openaq.org/v2/latest"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast" 

# Los Angeles Zones (Simulation)
LA_ZONES = {
    "Downtown": {"lat": 34.0522, "lon": -118.2437, "weight": 1.2, "name": "Downtown (DTLA)"},
    "Santa Monica": {"lat": 34.0195, "lon": -118.4912, "weight": 0.8, "name": "Santa Monica"},
    "San Fernando Valley": {"lat": 34.2798, "lon": -118.6706, "weight": 1.1, "name": "San Fernando Valley"},
    "Long Beach": {"lat": 33.7701, "lon": -118.1937, "weight": 1.0, "name": "Long Beach"},
}

# -----------------------------------------------------------------------------
# 1. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def get_aqi_status_and_color(pm25_value):
    """Determine AQI status and color from PM2.5 value."""
    try:
        pm25_float = float(pm25_value)
    except (ValueError, TypeError):
        return "Data unavailable", "gray", "❓"
    
    if pm25_float <= 12.0:
        return "Good", "green", "👍"
    elif pm25_float <= 35.4:
        return "Moderate", "yellow", "👌"
    elif pm25_float <= 55.4:
        return "Poor (Sensitive)", "orange", "⚠️"
    elif pm25_float <= 150.4:
        return "Unhealthy", "red", "🚨"
    elif pm25_float <= 250.4:
        return "Very Unhealthy", "purple", "❌"
    else:
        return "Hazardous", "maroon", "💀"

def get_risk_description(risk_level):
    """Provides description and advice based on risk level."""
    descriptions = {
        'low': ("Low", "Low risk. Enjoy outdoor activities. Monitor on high wind/fire days."),
        'medium': ("Medium", "Increased risk. Sensitive individuals (asthma, allergies) should limit intense exercise on 'Poor' days."),
        'high': ("High", "High sensitivity. Avoid all outdoor activities when air is 'Poor' or worse. Wear N95 mask if necessary."),
        'critical': ("Critical", "Serious respiratory issues. Stay indoors, use air purifier and consult doctor if symptoms occur."),
    }
    return descriptions.get(risk_level, descriptions['medium'])

def get_color_hex(color_name):
    """Convert color names to hex codes for Plotly/Streamlit."""
    colors = {
        "green": "#1ABC9C", "yellow": "#F7DC6F", "orange": "#E67E22",
        "red": "#E74C3C", "purple": "#8E44AD", "maroon": "#7B241C",
        "gray": "#95A5A6"
    }
    return colors.get(color_name, "#2C3E50")

# -----------------------------------------------------------------------------
# 2. DATA MANAGEMENT AND API (Simulation)
# -----------------------------------------------------------------------------

class DataFetcher:
    """Simulates real-time data and weather retrieval."""
    
    def fetch_multi_source_aqi(self, lat, lon):
        """Simulates AQI data retrieval (OpenAQ/Local Sensors)."""
        if random.random() < 0.7:
            pm25 = random.uniform(5, 50) 
            return [{'source': 'OpenAQ', 'pm25': pm25, 'lat': lat, 'lon': lon}]
        else:
            return None

    def fetch_advanced_weather(self, lat, lon):
        """Simulates advanced weather data retrieval."""
        weather = {
            'temperature': random.uniform(15, 30),
            'humidity': random.uniform(40, 70),
            'wind_speed': random.uniform(0, 15),
            'rain_1h': random.choice([0.0, 0.0, 0.0, 0.1, 0.5]),
            'is_sunny': random.choice([True, True, False, True]),
        }
        return weather

# -----------------------------------------------------------------------------
# 3. AQI PREDICTION ENGINE
# -----------------------------------------------------------------------------

class AdvancedAQIPredictor:
    """Simulated AQI prediction model with advanced factors."""
    
    def __init__(self):
        self.pm25_weight = 0.6
        self.no2_weight = 0.2
        self.o3_weight = 0.2
        
    def _calculate_seasonal_factors(self, dt: datetime):
        month = dt.month
        if month in [12, 1, 7, 8]:
            seasonal_factor = 1.15
        elif month in [2, 3, 9, 10]:
            seasonal_factor = 1.05
        else:
            seasonal_factor = 0.9
        return seasonal_factor

    def predict_advanced(self, current_data, weather_data, hours_ahead=1, confidence=False, reference_dt=datetime.now()):
        try:
            current_pm25 = current_data.get('pm25', 15)
            current_no2 = max(5, current_pm25 * 0.5 + random.uniform(-2, 2))
            current_o3 = max(10, current_pm25 * 0.8 + random.uniform(-5, 5))
            
            wind_factor = 1 - (weather_data.get('wind_speed', 5) / 50)
            rain_factor = 1 - (weather_data.get('rain_1h', 0) * 0.5)
            temp_factor = 1 + (weather_data.get('temperature', 20) / 400)
            
            target_dt = reference_dt + timedelta(hours=hours_ahead)
            hour = target_dt.hour
            
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                traffic_factor = 1.15
            else:
                traffic_factor = 0.95
                
            time_decay = 1 + (hours_ahead * 0.02)
            seasonal_factor = self._calculate_seasonal_factors(target_dt)
            
            combined_factor = (wind_factor * rain_factor * temp_factor * traffic_factor * seasonal_factor * time_decay)

            predicted_pm25 = (
                current_pm25 * self.pm25_weight +
                current_no2 * self.no2_weight +
                current_o3 * self.o3_weight
            ) * combined_factor
            
            predicted_pm25 = max(5, min(150, predicted_pm25 + random.uniform(-2, 2)))

            if confidence:
                uncertainty = np.log(hours_ahead + 1) * 2
                lower = max(0, predicted_pm25 - uncertainty)
                upper = predicted_pm25 + uncertainty
                return predicted_pm25, (lower, upper)

            return predicted_pm25

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return current_data.get('pm25', 20)

# -----------------------------------------------------------------------------
# 4. ALERT SYSTEMS AND CHATBOT
# -----------------------------------------------------------------------------

class SmartAlertSystem:
    """Generates recommendations based on AQI and user profile."""
    
    def __init__(self, user_profile, predicted_aqi):
        self.profile = user_profile
        self.predicted_aqi = predicted_aqi
        self.risk_level = user_profile['risk']
        self.aqi_status, _, _ = get_aqi_status_and_color(predicted_aqi)

    def generate_recommendations(self):
        recos = []
        
        recos.append(f"Predicted status: **{self.aqi_status}** ({self.predicted_aqi:.1f} µg/m³ PM2.5).")
        
        if self.risk_level in ['high', 'critical'] and self.predicted_aqi > 35:
            recos.append("😷 **High/Critical Sensitivity**: Avoid all intense outdoor physical activity. Consider wearing N95 mask.")
        elif self.risk_level == 'medium' and self.predicted_aqi > 55:
            recos.append("⚠️ **Medium Sensitivity**: Reduce prolonged outdoor efforts. If you experience symptoms, go inside immediately.")
        else:
            recos.append("✅ **General Advice**: Air is acceptable for most activities. Enjoy!")

        if "outdoor_sport" in self.profile['activities'] and self.predicted_aqi > 55:
            recos.append("🏃 **Sports**: Postpone outdoor running or cycling. Prefer gym or light indoor training.")
        
        if "walk" in self.profile['activities'] and self.predicted_aqi > 55:
            recos.append("🚶 **Commuting**: Limit long walks. Use public transport or electric bike/car if possible.")
            
        return recos

# -----------------------------------------------------------------------------
# 5. DISPLAY COMPONENTS
# -----------------------------------------------------------------------------

def display_zone_card(zone_data, predictor):
    pm25_value = zone_data['pm25']
    status, color, icon = get_aqi_status_and_color(pm25_value)
    color_hex = get_color_hex(color)
    
    forecast_pm25_12h = predictor.predict_advanced(
        {'pm25': pm25_value}, 
        zone_data['weather'], 
        hours_ahead=12,
        reference_dt=st.session_state.get('analysis_datetime', datetime.now())
    )
    forecast_status_12h, _, _ = get_aqi_status_and_color(forecast_pm25_12h)
    
    st.markdown(
        f"""
        <div style='
            background-color: white; 
            border-left: 10px solid {color_hex}; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        '>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h3 style='margin: 0; color: #333;'>{zone_data['zone']}</h3>
                <span style='font-size: 30px;'>{icon}</span>
            </div>
            <p style='margin-top: 5px; font-size: 1.1em;'>
                **Current PM2.5:** <span style='color: {color_hex}; font-weight: bold;'>{pm25_value:.1f} µg/m³</span> ({status})
            </p>
            <p style='margin: 0; font-size: 0.9em; color: #555;'>
                **Data source:** {zone_data.get('source', 'Unknown')}
            </p>
            <div style='margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;'>
                 <p style='margin: 0; font-size: 0.9em; font-weight: bold;'>Forecast (H+12):</p>
                 <p style='margin: 0; font-size: 1.0em;'>{forecast_pm25_12h:.1f} µg/m³ ({forecast_status_12h})</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    
def display_map_view(data_df):
    data_df['color'] = data_df['PM2.5'].apply(lambda x: get_color_hex(get_aqi_status_and_color(x)[1]))
    data_df['size'] = data_df['PM2.5'].apply(lambda x: 10 if x <= 12.0 else 15 if x <= 35.4 else 25)
    
    fig = px.scatter_mapbox(data_df, 
                            lat="Latitude", 
                            lon="Longitude", 
                            color="Status",
                            size="PM2.5",
                            hover_name="Zone",
                            hover_data={"PM2.5": ':.1f', "Status": True, "Source": True, "Latitude": False, "Longitude": False},
                            color_discrete_map={
                                "Good": get_color_hex("green"),
                                "Moderate": get_color_hex("yellow"),
                                "Poor (Sensitive)": get_color_hex("orange"),
                                "Unhealthy": get_color_hex("red"),
                                "Very Unhealthy": get_color_hex("purple"),
                                "Hazardous": get_color_hex("maroon"),
                                "Data unavailable": get_color_hex("gray")
                            },
                            zoom=9, 
                            height=600)
    
    fig.update_layout(mapbox_style="carto-positron", 
                      margin={"r":0,"t":0,"l":0,"b":0},
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    st.plotly_chart(fig, use_container_width=True)

def display_historic_and_forecast(zone_name, current_pm25, weather_data, predictor, reference_dt):
    forecast_points = []
    for h in range(1, 25):
        predicted_pm25, (lower, upper) = predictor.predict_advanced(
            {'pm25': current_pm25}, 
            weather_data, 
            hours_ahead=h, 
            confidence=True, 
            reference_dt=reference_dt
        )
        forecast_points.append({
            'Hour': reference_dt + timedelta(hours=h),
            'PM2.5': predicted_pm25,
            'Type': 'Forecast',
            'Lower': lower,
            'Upper': upper
        })

    historic_points = []
    for h in range(24, 0, -1):
        historic_dt = reference_dt - timedelta(hours=h)
        time_factor = 1.05 if 7 <= historic_dt.hour <= 9 or 16 <= historic_dt.hour <= 19 else 0.95
        simulated_pm25 = current_pm25 * time_factor + random.uniform(-5, 5) * (1 - (h / 24) * 0.5)
        simulated_pm25 = max(5, min(150, simulated_pm25))
        
        historic_points.append({
            'Hour': historic_dt,
            'PM2.5': simulated_pm25,
            'Type': 'Historical',
            'Lower': simulated_pm25,
            'Upper': simulated_pm25
        })

    current_point = [{
        'Hour': reference_dt,
        'PM2.5': current_pm25,
        'Type': 'Current',
        'Lower': current_pm25,
        'Upper': current_pm25
    }]
    
    data_df = pd.DataFrame(historic_points + current_point + forecast_points)

    fig = px.line(data_df, 
                  x='Hour', 
                  y='PM2.5', 
                  color='Type', 
                  title=f"Historical (Simulated) and PM2.5 Forecast for {zone_name}",
                  labels={'PM2.5': 'PM2.5 Concentration (µg/m³)', 'Hour': 'Date and Time'},
                  color_discrete_map={'Historical': '#2980B9', 'Current': '#F39C12', 'Forecast': '#E74C3C'})
    
    fig.add_scatter(x=data_df[data_df['Type'] == 'Forecast']['Hour'], 
                    y=data_df[data_df['Type'] == 'Forecast']['Upper'], 
                    mode='lines', 
                    name='Confidence Interval', 
                    line=dict(width=0),
                    showlegend=False)
    fig.add_scatter(x=data_df[data_df['Type'] == 'Forecast']['Hour'], 
                    y=data_df[data_df['Type'] == 'Forecast']['Lower'], 
                    mode='lines', 
                    name='Confidence Interval', 
                    line=dict(width=0),
                    fill='tonexty', 
                    fillcolor='rgba(231, 76, 60, 0.1)',
                    showlegend=False)
                    
    for limit, label, color in [(35.4, 'Poor', get_color_hex("orange")), (55.4, 'Unhealthy', get_color_hex("red"))]:
        fig.add_hline(y=limit, line_dash="dash", line_color=color, annotation_text=label, 
                      annotation_position="top right", annotation_font_color=color)

    fig.update_layout(xaxis_title="Hour", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. TAB FUNCTIONS
# -----------------------------------------------------------------------------

def tab_dashboard(fetcher, predictor, all_zone_data, df_map):
    """Tab 1: Dashboard with overview"""
    st.header("🟢 Dashboard - Overview")
    
    # Main metrics
    current_pm25_values = [zone['pm25'] for zone in all_zone_data]
    avg_pm25 = np.mean(current_pm25_values)
    max_pm25 = max(current_pm25_values)
    status, color, icon = get_aqi_status_and_color(avg_pm25)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average AQI", f"{avg_pm25:.1f} µg/m³", delta=f"{status}")
    with col2:
        st.metric("Most polluted zone", f"{max_pm25:.1f} µg/m³")
    with col3:
        st.metric("Monitored zones", len(LA_ZONES))
    with col4:
        st.metric("Global Status", status, delta="Stable" if avg_pm25 < 35 else "Degraded")
    
    # Map and summary
    col1, col2 = st.columns([7, 3])
    
    with col1:
        display_map_view(df_map)
        
    with col2:
        st.subheader("Zone Summary")
        for zone_data in all_zone_data:
            display_zone_card(zone_data, predictor)
    
    # Last 24h chart
    st.subheader("AQI Evolution - Last 24 hours")
    selected_zone_name = st.selectbox("Select Zone", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == selected_zone_name)
    
    display_historic_and_forecast(
        selected_zone_name, 
        selected_zone_data['pm25'], 
        selected_zone_data['weather'], 
        predictor, 
        st.session_state.get('analysis_datetime', datetime.now())
    )
    
    # Refresh button
    if st.button("🔄 Refresh data"):
        st.rerun()

def tab_historique(all_zone_data, predictor):
    """Tab 2: Historical trend analysis"""
    st.header("🔵 History and Trends")
    
    # Period and zone selectors
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Analysis period", ["7 days", "30 days", "3 months"])
    with col2:
        zone_hist = st.selectbox("Zone", [d['zone'] for d in all_zone_data])
    with col3:
        aggregation = st.selectbox("Aggregation", ["Hourly", "Daily", "Weekly"])
    
    # Simulated historical data generation
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_hist)
    end_date = st.session_state.get('analysis_datetime', datetime.now())
    
    if period == "7 days":
        days = 7
    elif period == "30 days":
        days = 30
    else:
        days = 90
    
    dates = [end_date - timedelta(days=x) for x in range(days, 0, -1)]
    historical_data = []
    
    for date in dates:
        # Seasonal and daily variation simulation
        base_pm25 = selected_zone_data['pm25']
        seasonal_factor = 1 + 0.1 * np.sin((date.month - 1) * np.pi / 6)  # Seasonal variation
        daily_variation = random.uniform(0.8, 1.2)  # Daily variation
        
        pm25_value = base_pm25 * seasonal_factor * daily_variation
        pm25_value = max(5, min(150, pm25_value))
        
        historical_data.append({
            'Date': date,
            'PM2.5': pm25_value,
            'Zone': zone_hist
        })
    
    df_hist = pd.DataFrame(historical_data)
    
    # Historical chart
    fig = px.line(df_hist, x='Date', y='PM2.5', 
                  title=f"AQI History - {zone_hist} ({period})",
                  labels={'PM2.5': 'PM2.5 Concentration (µg/m³)', 'Date': 'Date'})
    
    # Threshold lines
    for limit, label, color in [(35.4, 'Poor', get_color_hex("orange")), 
                               (55.4, 'Unhealthy', get_color_hex("red"))]:
        fig.add_hline(y=limit, line_dash="dash", line_color=color, 
                     annotation_text=label, annotation_position="top right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{df_hist['PM2.5'].mean():.1f} µg/m³")
    with col2:
        st.metric("Maximum", f"{df_hist['PM2.5'].max():.1f} µg/m³")
    with col3:
        st.metric("Minimum", f"{df_hist['PM2.5'].min():.1f} µg/m³")
    with col4:
        trend = "Stable" if abs(df_hist['PM2.5'].pct_change().mean()) < 0.1 else "Rising" if df_hist['PM2.5'].pct_change().mean() > 0 else "Declining"
        st.metric("Trend", trend)
    
    # Data download
    csv = df_hist.to_csv(index=False)
    st.download_button(
        label="📥 Download historical data (CSV)",
        data=csv,
        file_name=f"aqi_history_{zone_hist}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def tab_previsions(all_zone_data, predictor):
    """Tab 3: Future forecasts"""
    st.header("🟣 AQI Forecasts")
    
    # Zone selector
    zone_prev = st.selectbox("Zone for forecasts", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_prev)
    
    # Detailed forecasts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        horizon = st.slider("Forecast horizon (hours)", 1, 72, 24)
        
        # Forecast generation
        forecast_data = []
        for h in range(1, horizon + 1):
            predicted_pm25 = predictor.predict_advanced(
                {'pm25': selected_zone_data['pm25']}, 
                selected_zone_data['weather'], 
                hours_ahead=h,
                reference_dt=st.session_state.get('analysis_datetime', datetime.now())
            )
            status, color, icon = get_aqi_status_and_color(predicted_pm25)
            
            forecast_time = st.session_state.get('analysis_datetime', datetime.now()) + timedelta(hours=h)
            forecast_data.append({
                'Hour': forecast_time,
                'PM2.5': predicted_pm25,
                'Status': status,
                'Color': color,
                'Icon': icon
            })
        
        df_forecast = pd.DataFrame(forecast_data)
        
        # Forecast chart
        fig = px.line(df_forecast, x='Hour', y='PM2.5',
                      title=f"AQI Forecasts - {zone_prev}",
                      labels={'PM2.5': 'PM2.5 Concentration (µg/m³)', 'Hour': 'Date and Time'})
        
        # Color zones by status
        for i in range(len(df_forecast)-1):
            status = df_forecast.iloc[i]['Status']
            color = get_color_hex(df_forecast.iloc[i]['Color'])
            fig.add_vrect(x0=df_forecast.iloc[i]['Hour'], 
                         x1=df_forecast.iloc[i+1]['Hour'],
                         fillcolor=color, opacity=0.2, line_width=0)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Weather Indicators")
        weather = selected_zone_data['weather']
        
        st.metric("🌡️ Temperature", f"{weather['temperature']:.1f}°C")
        st.metric("💧 Humidity", f"{weather['humidity']:.1f}%")
        st.metric("💨 Wind speed", f"{weather['wind_speed']:.1f} km/h")
        st.metric("🌧️ Rain (1h)", f"{weather['rain_1h']:.1f} mm")
        
        st.subheader("🚨 Risk Zones")
        risky_zones = []
        for zone in all_zone_data:
            forecast_24h = predictor.predict_advanced(
                {'pm25': zone['pm25']}, 
                zone['weather'], 
                hours_ahead=24,
                reference_dt=st.session_state.get('analysis_datetime', datetime.now())
            )
            if forecast_24h > 35.4:  # "Poor" threshold
                risky_zones.append((zone['zone'], forecast_24h))
        
        if risky_zones:
            for zone_name, pm25 in risky_zones:
                status, color, icon = get_aqi_status_and_color(pm25)
                st.warning(f"{icon} {zone_name}: {pm25:.1f} µg/m³ ({status})")
        else:
            st.success("✅ No high-risk zones forecasted")
        
        # "What if" simulation
        st.subheader("🔬 'What If' Simulation")
        new_temp = st.slider("Simulated temperature (°C)", 10, 40, int(weather['temperature']))
        new_wind = st.slider("Simulated wind (km/h)", 0, 30, int(weather['wind_speed']))
        
        if st.button("Simulate impact"):
            simulated_weather = weather.copy()
            simulated_weather['temperature'] = new_temp
            simulated_weather['wind_speed'] = new_wind
            
            original_24h = predictor.predict_advanced(
                {'pm25': selected_zone_data['pm25']}, 
                weather, 
                hours_ahead=24
            )
            simulated_24h = predictor.predict_advanced(
                {'pm25': selected_zone_data['pm25']}, 
                simulated_weather, 
                hours_ahead=24
            )
            
            difference = simulated_24h - original_24h
            st.info(f"Simulated impact: {difference:+.1f} µg/m³")

def tab_profil_utilisateur():
    """Tab 4: User profile"""
    st.header("🟠 User Profile")
    
    with st.form("user_profile_form"):
        st.subheader("📝 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.selectbox("Age", ["<18", "18-30", "31-50", "51-65", "65+"])
            conditions = st.multiselect("Health conditions", 
                                      ["Asthma", "Allergies", "Heart problems", "None"])
        
        with col2:
            profession = st.selectbox("Profession", 
                                    ["Office", "Outdoor", "Student", "Retired", "Other"])
            activities = st.multiselect("Regular activities",
                                     ["Intense sports", "Walking", "Cycling", "Gardening", "None"])
        
        # Form submission
        submitted = st.form_submit_button("💾 Save profile")
        
        if submitted:
            st.session_state['user_profile_details'] = {
                'age': age,
                'conditions': conditions,
                'profession': profession,
                'activities': activities
            }
            st.success("✅ Profile saved successfully!")
    
    # Risk summary display
    if 'user_profile_details' in st.session_state:
        st.subheader("📊 Your Personal Risk Summary")
        
        profile = st.session_state['user_profile_details']
        risk_factors = 0
        
        # Risk factor calculation
        if profile['age'] in ['<18', '65+']:
            risk_factors += 1
        if any(cond in profile['conditions'] for cond in ['Asthma', 'Heart problems']):
            risk_factors += 2
        if profile['profession'] == 'Outdoor':
            risk_factors += 1
        if 'Intense sports' in profile['activities']:
            risk_factors += 1
        
        # Risk level determination
        if risk_factors >= 3:
            risk_level = "High"
            color = "red"
            advice = "Monitor air quality closely. Avoid outdoor activities when AQI is poor or worse."
        elif risk_factors >= 2:
            risk_level = "Moderate"
            color = "orange"
            advice = "Be cautious on polluted days. Reduce intense outdoor activities."
        else:
            risk_level = "Low"
            color = "green"
            advice = "You are less sensitive to pollution. Continue to enjoy your normal activities."
        
        st.markdown(f"<h3 style='color:{color}'>Risk level: {risk_level}</h3>", unsafe_allow_html=True)
        st.info(f"💡 **Personalized advice**: {advice}")
        
        # Activity suggestions based on current AQI
        st.subheader("🎯 Activity Suggestions")
        
        # Current AQI simulation (in practice, use real data)
        current_aqi = 25  # Simulated value
        
        if current_aqi <= 35:
            st.success("**✅ Excellent conditions** - Ideal for all outdoor activities")
            st.write("• Intense outdoor sports")
            st.write("• Hiking and cycling")
            st.write("• Outdoor picnics")
        elif current_aqi <= 55:
            st.warning("**⚠️ Moderate conditions** - Adapt your activities")
            st.write("• Moderate sports acceptable")
            st.write("• Avoid prolonged efforts")
            st.write("• Sensitive people: limit time outdoors")
        else:
            st.error("**🚨 Poor conditions** - Prefer indoor activities")
            st.write("• Postpone sports activities")
            st.write("• Limit non-essential travel")
            st.write("• Use air purifier indoors")

def tab_analyse_zone(all_zone_data):
    """Tab 5: Detailed zone analysis"""
    st.header("⚫ Zone Analysis")
    
    # Zone selector
    zone_analyze = st.selectbox("Select a zone", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_analyze)
    
    # Zone metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current AQI", f"{selected_zone_data['pm25']:.1f} µg/m³")
    with col2:
        # Min/max AQI calculation (simulation)
        aqi_min = max(5, selected_zone_data['pm25'] * 0.7)
        aqi_max = min(150, selected_zone_data['pm25'] * 1.3)
        st.metric("Min AQI (24h)", f"{aqi_min:.1f} µg/m³")
    with col3:
        st.metric("Max AQI (24h)", f"{aqi_max:.1f} µg/m³")
    with col4:
        st.metric("Average AQI", f"{(aqi_min + aqi_max) / 2:.1f} µg/m³")
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution (simulation)
        hours = list(range(24))
        aqi_hourly = [selected_zone_data['pm25'] * (1 + 0.3 * np.sin(h/24 * 2 * np.pi)) for h in hours]
        
        fig_hourly = px.line(x=hours, y=aqi_hourly,
                            title=f"Typical Hourly Profile - {zone_analyze}",
                            labels={'x': 'Time of day', 'y': 'AQI (µg/m³)'})
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Comparison with other zones
        comparison_zones = st.multiselect("Zones to compare",
                                         [d['zone'] for d in all_zone_data if d['zone'] != zone_analyze],
                                         default=[d['zone'] for d in all_zone_data if d['zone'] != zone_analyze][:2])
        
        if comparison_zones:
            comparison_data = [selected_zone_data]
            for zone in all_zone_data:
                if zone['zone'] in comparison_zones:
                    comparison_data.append(zone)
            
            fig_comparison = px.bar(
                x=[d['zone'] for d in comparison_data],
                y=[d['pm25'] for d in comparison_data],
                title="AQI Comparison Between Zones",
                labels={'x': 'Zone', 'y': 'AQI (µg/m³)'}
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed trends
    st.subheader("📈 Detailed Trends")
    
    # Trend data simulation
    days = 30
    trend_dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
    trend_data = []
    
    for date in trend_dates:
        variation = random.uniform(0.8, 1.2)
        aqi_day = selected_zone_data['pm25'] * variation
        trend_data.append({'Date': date, 'AQI': aqi_day})
    
    df_trend = pd.DataFrame(trend_data)
    
    fig_trend = px.line(df_trend, x='Date', y='AQI',
                      title=f"AQI Trend - {zone_analyze} (30 days)",
                      labels={'AQI': 'PM2.5 Concentration (µg/m³)'})
    
    # Trend line addition
    z = np.polyfit(range(len(df_trend)), df_trend['AQI'], 1)
    p = np.poly1d(z)
    fig_trend.add_scatter(x=df_trend['Date'], y=p(range(len(df_trend))),
                       mode='lines', name='Trend', line=dict(dash='dash'))
    
    st.plotly_chart(fig_trend, use_container_width=True)

def tab_facteurs_pollution(all_zone_data):
    """Tab 6: Pollution factors"""
    st.header("⚪ Pollution Factors")
    
    selected_zone = st.selectbox("Analysis zone", [d['zone'] for d in all_zone_data])
    selected_data = next(d for d in all_zone_data if d['zone'] == selected_zone)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Pollution Sources")
        
        # Pollution sources simulation
        sources = {
            'Traffic': random.uniform(20, 40),
            'Industry': random.uniform(10, 25),
            'Heating': random.uniform(5, 15),
            'Agriculture': random.uniform(8, 18),
            'Natural': random.uniform(5, 12),
            'Other': random.uniform(10, 20)
        }
        
        # Normalization to 100%
        total = sum(sources.values())
        sources = {k: (v/total)*100 for k, v in sources.items()}
        
        fig_sources = px.pie(values=list(sources.values()), names=list(sources.keys()),
                           title="Pollution Source Contribution")
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Weather Correlations")
        
        # Correlation data simulation
        temperature = np.random.normal(20, 5, 100)
        humidity = np.random.normal(60, 15, 100)
        aqi_simulated = selected_data['pm25'] + temperature * 0.5 - humidity * 0.3 + np.random.normal(0, 5, 100)
        
        fig_corr_temp = px.scatter(x=temperature, y=aqi_simulated,
                                 title="AQI vs Temperature Correlation",
                                 labels={'x': 'Temperature (°C)', 'y': 'AQI (µg/m³)'})
        fig_corr_temp.update_layout(showlegend=False)
        st.plotly_chart(fig_corr_temp, use_container_width=True)
        
        fig_corr_hum = px.scatter(x=humidity, y=aqi_simulated,
                                title="AQI vs Humidity Correlation",
                                labels={'x': 'Humidity (%)', 'y': 'AQI (µg/m³)'})
        fig_corr_hum.update_layout(showlegend=False)
        st.plotly_chart(fig_corr_hum, use_container_width=True)
    
    # Tips to reduce impact
    st.subheader("💡 Reduce Your Impact")
    
    tips = [
        "🚗 Use public transport or carpooling",
        "🚲 Prefer cycling for short trips",
        "💡 Turn off lights and unused appliances",
        "🌱 Reduce meat consumption",
        "🛒 Buy local and seasonal products",
        "♻️ Recycle and compost your waste",
        "🏠 Improve home insulation",
        "🌳 Plant trees and vegetation"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")

def tab_rapports_export(all_zone_data):
    """Tab 7: Reports and export"""
    st.header("🟩 Reports & Export")
    
    # Automatic report generation
    st.subheader("📄 Automatic Report")
    
    # Global statistics calculation
    aqi_values = [zone['pm25'] for zone in all_zone_data]
    avg_aqi = np.mean(aqi_values)
    max_aqi = max(aqi_values)
    max_zone = all_zone_data[np.argmax(aqi_values)]['zone']
    global_status, _, _ = get_aqi_status_and_color(avg_aqi)
    
    # Text report
    report = f"""
    ## 📊 Air Quality Report - {datetime.now().strftime('%m/%d/%Y')}
    
    ### Executive Summary
    - **Overall quality**: {global_status}
    - **Average AQI**: {avg_aqi:.1f} µg/m³
    - **Most affected zone**: {max_zone} ({max_aqi:.1f} µg/m³)
    - **Number of monitored zones**: {len(all_zone_data)}
    
    ### Zone Details
    """
    
    for zone in all_zone_data:
        status, _, icon = get_aqi_status_and_color(zone['pm25'])
        report += f"- **{zone['zone']}**: {zone['pm25']:.1f} µg/m³ {icon} ({status})\n"
    
    report += """
    ### Recommendations
    - Monitor high-risk zones
    - Adapt activities according to air quality
    - Check forecasts to plan your outings
    """
    
    st.markdown(report)
    
    # Export options
    st.subheader("📤 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        df_export = pd.DataFrame([{
            'Zone': zone['zone'],
            'AQI': zone['pm25'],
            'Status': get_aqi_status_and_color(zone['pm25'])[0],
            'Latitude': zone['lat'],
            'Longitude': zone['lon'],
            'Source': zone.get('source', 'Simulated'),
            'Date': datetime.now()
        } for zone in all_zone_data])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"aqi_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Simulated PDF export
        if st.button("📄 Generate PDF"):
            st.info("PDF functionality under development...")
            st.success("✅ PDF report generated successfully! (simulation)")
    
    # Report visualizations
    st.subheader("📈 Report Visualizations")
    
    # Comparative chart
    fig_report = px.bar(
        x=[zone['zone'] for zone in all_zone_data],
        y=[zone['pm25'] for zone in all_zone_data],
        title="AQI Comparison Between All Zones",
        labels={'x': 'Zone', 'y': 'AQI (µg/m³)'},
        color=[get_aqi_status_and_color(zone['pm25'])[1] for zone in all_zone_data]
    )
    st.plotly_chart(fig_report, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. NAVIGATION AND SESSION STATE
# -----------------------------------------------------------------------------

def create_navigation():
    """Creates advanced sidebar navigation with date/time selection and user profile."""
    if 'user_profile' not in st.session_state:
        st.session_state['user_profile'] = {'risk': 'medium', 'activities': ['outdoor_sport', 'walk']}
    if 'analysis_datetime' not in st.session_state:
        st.session_state['analysis_datetime'] = datetime.now().replace(minute=0, second=0, microsecond=0)
    if 'selected_zone' not in st.session_state:
        st.session_state['selected_zone'] = LA_ZONES["Downtown"]
    
    with st.sidebar:
        st.title("🎛️ Analysis Settings")
        
        # Time selection
        st.subheader("⏱️ Analysis Period")
        
        current_dt = datetime.now()
        analysis_dt = st.session_state['analysis_datetime']

        selected_date = st.date_input(
            "Analysis date",
            value=analysis_dt.date(),
            min_value=current_dt.date() - timedelta(days=7),
            max_value=current_dt.date() + timedelta(days=3)
        )
        
        selected_time = st.time_input("Analysis time", value=analysis_dt.time())
        
        new_analysis_dt = datetime.combine(selected_date, selected_time)
        if new_analysis_dt != st.session_state['analysis_datetime']:
             st.session_state['analysis_datetime'] = new_analysis_dt
             st.rerun()

        st.markdown("---")
        
        # User profile
        st.subheader("👤 Your Risk Profile")
        
        risk_level = st.selectbox(
            "Pollution Sensitivity Level",
            options=['low', 'medium', 'high', 'critical'],
            format_func=lambda x: get_risk_description(x)[0],
            index=['low', 'medium', 'high', 'critical'].index(st.session_state['user_profile']['risk'])
        )
        
        activities = st.multiselect(
            "Planned Outdoor Activities",
            options=['outdoor_sport', 'walk', 'bike', 'garden'],
            default=st.session_state['user_profile']['activities'],
            format_func=lambda x: {"outdoor_sport": "Intense Sports", "walk": "Walking", "bike": "Cycling", "garden": "Gardening/Leisure"}.get(x, x)
        )
        
        st.session_state['user_profile'] = {'risk': risk_level, 'activities': activities}
        
        st.markdown(f"**Advice**: *{get_risk_description(risk_level)[1]}*")

# -----------------------------------------------------------------------------
# 8. MAIN APPLICATION WITH TABS
# -----------------------------------------------------------------------------

def main_application():
    """Main Streamlit application controller with tabs."""
    
    # Class initialization
    fetcher = DataFetcher()
    predictor = AdvancedAQIPredictor()
    
    st.title("Guardian AI 🌍 Advanced Air Quality (AQI) Forecasts")
    
    # Get reference time
    analysis_dt = st.session_state.get('analysis_datetime', datetime.now())
    is_historic = analysis_dt < datetime.now() - timedelta(minutes=60)
    
    if is_historic:
        st.header(f"Historical Mode: Data for **{analysis_dt.strftime('%m/%d/%Y at %H:%M')}**")
    elif analysis_dt > datetime.now() + timedelta(minutes=10):
        st.header(f"Forecast Mode: Data for **{analysis_dt.strftime('%m/%d/%Y at %H:%M')}**")
    else:
        st.header(f"Current Mode: Data for **{analysis_dt.strftime('%m/%d/%Y at %H:%M')}**")
    
    st.markdown("---")
    
    # Data retrieval
    with st.spinner('🛰️ Retrieving and harmonizing satellite and ground data...'):
        all_zone_data = []
        df_map_data = []

        for zone_name, zone_info in LA_ZONES.items():
            
            weather_data = fetcher.fetch_advanced_weather(zone_info['lat'], zone_info['lon'])
            avg_pm25 = None
            source = "Simulated (Model)"

            if not is_historic and analysis_dt < datetime.now() + timedelta(minutes=10):
                zone_data_from_api = fetcher.fetch_multi_source_aqi(zone_info['lat'], zone_info['lon'])
                
                if zone_data_from_api:
                    pm25_values = [d['pm25'] for d in zone_data_from_api if d.get('pm25') is not None]
                    if pm25_values:
                        avg_pm25 = np.mean(pm25_values) * zone_info['weight']
                        source = "Real (OpenAQ)"

            if avg_pm25 is None:
                time_of_day_factor = 1.2 if 7 <= analysis_dt.hour <= 9 or 16 <= analysis_dt.hour <= 19 else 0.8
                delta_seconds = (analysis_dt - datetime.now()).total_seconds()
                delta_hours = delta_seconds / 3600
                trend_factor = 1.0 + (delta_hours * 0.01)
                
                pm25_simulated = (25 * time_of_day_factor * trend_factor + random.uniform(-5, 5)) * zone_info['weight']
                avg_pm25 = max(5, min(150, pm25_simulated))
                
                if not is_historic and analysis_dt < datetime.now() + timedelta(minutes=10):
                    st.sidebar.info(f"⚠️ **{zone_info['name']}**: No real data available. Simulation used.")
                
                if is_historic:
                     source = "Simulated Historical"
                elif analysis_dt > datetime.now():
                    source = "Simulated Forecast"

            avg_pm25 = float(avg_pm25) if avg_pm25 is not None else 25.0

            status, _, _ = get_aqi_status_and_color(avg_pm25)
            
            data_entry = {
                'zone': zone_info['name'],
                'pm25': avg_pm25,
                'weather': weather_data,
                'lat': zone_info['lat'],
                'lon': zone_info['lon'],
                'source': source
            }
            all_zone_data.append(data_entry)
            
            df_map_data.append({
                'Zone': zone_info['name'],
                'PM2.5': avg_pm25,
                'Status': status,
                'Latitude': zone_info['lat'],
                'Longitude': zone_info['lon'],
                'Source': source
            })

    df_map = pd.DataFrame(df_map_data)
    
    # Selected zone update
    zone_names = [d['zone'] for d in all_zone_data]
    selected_zone_name = st.sidebar.selectbox("Select Detailed Analysis Zone", zone_names)
    st.session_state['selected_zone'] = next(info for name, info in LA_ZONES.items() if info['name'] == selected_zone_name)
    
    # Tab creation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🟢 Dashboard", 
        "🔵 History", 
        "🟣 Forecasts", 
        "🟠 Profile", 
        "⚫ Zone Analysis", 
        "⚪ Pollution Factors", 
        "🟩 Reports"
    ])
    
    # Tab content
    with tab1:
        tab_dashboard(fetcher, predictor, all_zone_data, df_map)
    
    with tab2:
        tab_historique(all_zone_data, predictor)
    
    with tab3:
        tab_previsions(all_zone_data, predictor)
    
    with tab4:
        tab_profil_utilisateur()
    
    with tab5:
        tab_analyse_zone(all_zone_data)
    
    with tab6:
        tab_facteurs_pollution(all_zone_data)
    
    with tab7:
        tab_rapports_export(all_zone_data)

# -----------------------------------------------------------------------------
# 9. ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    create_navigation()
    main_application()
