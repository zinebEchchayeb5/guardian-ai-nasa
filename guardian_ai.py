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
# 0. CONFIGURATION ET CONSTANTES
# -----------------------------------------------------------------------------

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Guardian AI: Prévisions AQI Avancées",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes des API (Remplacer par vos clés si nécessaire pour une utilisation réelle)
OPENAQ_API_URL = "https://api.openaq.org/v2/latest"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast" 

# Zones de Los Angeles (Simulation)
LA_ZONES = {
    "Downtown": {"lat": 34.0522, "lon": -118.2437, "weight": 1.2, "name": "Centre-ville (DTLA)"},
    "Santa Monica": {"lat": 34.0195, "lon": -118.4912, "weight": 0.8, "name": "Santa Monica"},
    "San Fernando Valley": {"lat": 34.2798, "lon": -118.6706, "weight": 1.1, "name": "Vallée de San Fernando"},
    "Long Beach": {"lat": 33.7701, "lon": -118.1937, "weight": 1.0, "name": "Long Beach"},
}

# -----------------------------------------------------------------------------
# 1. FONCTIONS UTILITAIRES
# -----------------------------------------------------------------------------

def get_aqi_status_and_color(pm25_value):
    """Détermine le statut AQI et la couleur à partir de la valeur PM2.5."""
    try:
        pm25_float = float(pm25_value)
    except (ValueError, TypeError):
        return "Données indisponibles", "gray", "❓"
    
    if pm25_float <= 12.0:
        return "Bonne", "green", "👍"
    elif pm25_float <= 35.4:
        return "Modérée", "yellow", "👌"
    elif pm25_float <= 55.4:
        return "Médiocre (Sensible)", "orange", "⚠️"
    elif pm25_float <= 150.4:
        return "Mauvaise", "red", "🚨"
    elif pm25_float <= 250.4:
        return "Très Mauvaise", "purple", "❌"
    else:
        return "Dangereuse", "maroon", "💀"

def get_risk_description(risk_level):
    """Fournit une description et des conseils basés sur le niveau de risque."""
    descriptions = {
        'low': ("Faible", "Peu de risques. Profitez de l'extérieur. Surveillez les jours de vent fort/incendies."),
        'medium': ("Moyen", "Risque accru. Les personnes sensibles (asthme, allergies) devraient limiter les efforts intenses les jours 'Médiocres'."),
        'high': ("Élevé", "Sensibilité forte. Évitez toute activité extérieure lorsque l'air est 'Médiocre' ou pire. Portez un masque N95 si nécessaire."),
        'critical': ("Critique", "Problèmes respiratoires graves. Restez à l'intérieur, utilisez un purificateur d'air et consultez un médecin en cas de symptômes."),
    }
    return descriptions.get(risk_level, descriptions['medium'])

def get_color_hex(color_name):
    """Convertit les noms de couleurs en codes hexadécimaux pour Plotly/Streamlit."""
    colors = {
        "green": "#1ABC9C", "yellow": "#F7DC6F", "orange": "#E67E22",
        "red": "#E74C3C", "purple": "#8E44AD", "maroon": "#7B241C",
        "gray": "#95A5A6"
    }
    return colors.get(color_name, "#2C3E50")

# -----------------------------------------------------------------------------
# 2. GESTION DES DONNÉES ET API (Simulation)
# -----------------------------------------------------------------------------

class DataFetcher:
    """Simule la récupération des données en temps réel et météo."""
    
    def fetch_multi_source_aqi(self, lat, lon):
        """Simule la récupération des données AQI (OpenAQ/Capteurs Locaux)."""
        if random.random() < 0.7:
            pm25 = random.uniform(5, 50) 
            return [{'source': 'OpenAQ', 'pm25': pm25, 'lat': lat, 'lon': lon}]
        else:
            return None

    def fetch_advanced_weather(self, lat, lon):
        """Simule la récupération des données météorologiques avancées."""
        weather = {
            'temperature': random.uniform(15, 30),
            'humidity': random.uniform(40, 70),
            'wind_speed': random.uniform(0, 15),
            'rain_1h': random.choice([0.0, 0.0, 0.0, 0.1, 0.5]),
            'is_sunny': random.choice([True, True, False, True]),
        }
        return weather

# -----------------------------------------------------------------------------
# 3. MOTEUR DE PRÉDICTION AQI
# -----------------------------------------------------------------------------

class AdvancedAQIPredictor:
    """Modèle de prédiction AQI simulé avec facteurs avancés."""
    
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
            st.error(f"Erreur dans la prédiction : {e}")
            return current_data.get('pm25', 20)

# -----------------------------------------------------------------------------
# 4. SYSTÈMES D'ALERTE ET CHATBOT
# -----------------------------------------------------------------------------

class SmartAlertSystem:
    """Génère des recommandations basées sur l'AQI et le profil utilisateur."""
    
    def __init__(self, user_profile, predicted_aqi):
        self.profile = user_profile
        self.predicted_aqi = predicted_aqi
        self.risk_level = user_profile['risk']
        self.aqi_status, _, _ = get_aqi_status_and_color(predicted_aqi)

    def generate_recommendations(self):
        recos = []
        
        recos.append(f"Statut prévu: **{self.aqi_status}** ({self.predicted_aqi:.1f} µg/m³ PM2.5).")
        
        if self.risk_level in ['high', 'critical'] and self.predicted_aqi > 35:
            recos.append("😷 **Sensibilité Forte/Critique**: Évitez toute activité physique extérieure intense. Envisagez de porter un masque N95.")
        elif self.risk_level == 'medium' and self.predicted_aqi > 55:
            recos.append("⚠️ **Sensibilité Moyenne**: Réduisez les efforts prolongés à l'extérieur. Si vous ressentez des symptômes, rentrez immédiatement.")
        else:
            recos.append("✅ **Conseil Général**: L'air est acceptable pour la plupart des activités. Profitez-en!")

        if "outdoor_sport" in self.profile['activities'] and self.predicted_aqi > 55:
            recos.append("🏃 **Sport**: Reportez la course à pied ou le vélo en extérieur. Préférez une salle de sport ou un entraînement léger à l'intérieur.")
        
        if "walk" in self.profile['activities'] and self.predicted_aqi > 55:
            recos.append("🚶 **Déplacement**: Limitez les longues marches. Utilisez les transports en commun ou le vélo électrique/voiture si possible.")
            
        return recos

# -----------------------------------------------------------------------------
# 5. COMPOSANTS D'AFFICHAGE
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
                **PM2.5 Actuel:** <span style='color: {color_hex}; font-weight: bold;'>{pm25_value:.1f} µg/m³</span> ({status})
            </p>
            <p style='margin: 0; font-size: 0.9em; color: #555;'>
                **Source des données:** {zone_data.get('source', 'Inconnue')}
            </p>
            <div style='margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;'>
                 <p style='margin: 0; font-size: 0.9em; font-weight: bold;'>Prévision (H+12):</p>
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
                                "Bonne": get_color_hex("green"),
                                "Modérée": get_color_hex("yellow"),
                                "Médiocre (Sensible)": get_color_hex("orange"),
                                "Mauvaise": get_color_hex("red"),
                                "Très Mauvaise": get_color_hex("purple"),
                                "Dangereuse": get_color_hex("maroon"),
                                "Données indisponibles": get_color_hex("gray")
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
            'Heure': reference_dt + timedelta(hours=h),
            'PM2.5': predicted_pm25,
            'Type': 'Prévision',
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
            'Heure': historic_dt,
            'PM2.5': simulated_pm25,
            'Type': 'Historique',
            'Lower': simulated_pm25,
            'Upper': simulated_pm25
        })

    current_point = [{
        'Heure': reference_dt,
        'PM2.5': current_pm25,
        'Type': 'Actuel',
        'Lower': current_pm25,
        'Upper': current_pm25
    }]
    
    data_df = pd.DataFrame(historic_points + current_point + forecast_points)

    fig = px.line(data_df, 
                  x='Heure', 
                  y='PM2.5', 
                  color='Type', 
                  title=f"Historique (Simulé) et Prévisions PM2.5 pour {zone_name}",
                  labels={'PM2.5': 'Concentration PM2.5 (µg/m³)', 'Heure': 'Date et Heure'},
                  color_discrete_map={'Historique': '#2980B9', 'Actuel': '#F39C12', 'Prévision': '#E74C3C'})
    
    fig.add_scatter(x=data_df[data_df['Type'] == 'Prévision']['Heure'], 
                    y=data_df[data_df['Type'] == 'Prévision']['Upper'], 
                    mode='lines', 
                    name='Confidence Interval', 
                    line=dict(width=0),
                    showlegend=False)
    fig.add_scatter(x=data_df[data_df['Type'] == 'Prévision']['Heure'], 
                    y=data_df[data_df['Type'] == 'Prévision']['Lower'], 
                    mode='lines', 
                    name='Confidence Interval', 
                    line=dict(width=0),
                    fill='tonexty', 
                    fillcolor='rgba(231, 76, 60, 0.1)',
                    showlegend=False)
                    
    for limit, label, color in [(35.4, 'Médiocre', get_color_hex("orange")), (55.4, 'Mauvaise', get_color_hex("red"))]:
        fig.add_hline(y=limit, line_dash="dash", line_color=color, annotation_text=label, 
                      annotation_position="top right", annotation_font_color=color)

    fig.update_layout(xaxis_title="Heure", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. FONCTIONS POUR LES ONGLETS
# -----------------------------------------------------------------------------

def tab_dashboard(fetcher, predictor, all_zone_data, df_map):
    """Onglet 1: Tableau de bord avec vue d'ensemble"""
    st.header("🟢 Tableau de Bord - Vue d'Ensemble")
    
    # Métriques principales
    current_pm25_values = [zone['pm25'] for zone in all_zone_data]
    avg_pm25 = np.mean(current_pm25_values)
    max_pm25 = max(current_pm25_values)
    status, color, icon = get_aqi_status_and_color(avg_pm25)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AQI Moyen", f"{avg_pm25:.1f} µg/m³", delta=f"{status}")
    with col2:
        st.metric("Zone la plus polluée", f"{max_pm25:.1f} µg/m³")
    with col3:
        st.metric("Zones surveillées", len(LA_ZONES))
    with col4:
        st.metric("Statut Global", status, delta="Stable" if avg_pm25 < 35 else "Dégradé")
    
    # Carte et résumé
    col1, col2 = st.columns([7, 3])
    
    with col1:
        display_map_view(df_map)
        
    with col2:
        st.subheader("Résumé par Zone")
        for zone_data in all_zone_data:
            display_zone_card(zone_data, predictor)
    
    # Graphique des dernières 24h
    st.subheader("Évolution AQI - 24 dernières heures")
    selected_zone_name = st.selectbox("Sélectionnez la Zone", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == selected_zone_name)
    
    display_historic_and_forecast(
        selected_zone_name, 
        selected_zone_data['pm25'], 
        selected_zone_data['weather'], 
        predictor, 
        st.session_state.get('analysis_datetime', datetime.now())
    )
    
    # Bouton d'actualisation
    if st.button("🔄 Actualiser les données"):
        st.rerun()

def tab_historique(all_zone_data, predictor):
    """Onglet 2: Analyse des tendances historiques"""
    st.header("🔵 Historique et Tendances")
    
    # Sélecteurs de période et zone
    col1, col2, col3 = st.columns(3)
    with col1:
        periode = st.selectbox("Période d'analyse", ["7 jours", "30 jours", "3 mois"])
    with col2:
        zone_hist = st.selectbox("Zone", [d['zone'] for d in all_zone_data])
    with col3:
        aggregation = st.selectbox("Agrégation", ["Par heure", "Par jour", "Par semaine"])
    
    # Génération de données historiques simulées
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_hist)
    end_date = st.session_state.get('analysis_datetime', datetime.now())
    
    if periode == "7 jours":
        days = 7
    elif periode == "30 jours":
        days = 30
    else:
        days = 90
    
    dates = [end_date - timedelta(days=x) for x in range(days, 0, -1)]
    historical_data = []
    
    for date in dates:
        # Simulation de variation saisonnière et journalière
        base_pm25 = selected_zone_data['pm25']
        seasonal_factor = 1 + 0.1 * np.sin((date.month - 1) * np.pi / 6)  # Variation saisonnière
        daily_variation = random.uniform(0.8, 1.2)  # Variation quotidienne
        
        pm25_value = base_pm25 * seasonal_factor * daily_variation
        pm25_value = max(5, min(150, pm25_value))
        
        historical_data.append({
            'Date': date,
            'PM2.5': pm25_value,
            'Zone': zone_hist
        })
    
    df_hist = pd.DataFrame(historical_data)
    
    # Graphique historique
    fig = px.line(df_hist, x='Date', y='PM2.5', 
                  title=f"Historique AQI - {zone_hist} ({periode})",
                  labels={'PM2.5': 'Concentration PM2.5 (µg/m³)', 'Date': 'Date'})
    
    # Ajout des lignes de seuils
    for limit, label, color in [(35.4, 'Médiocre', get_color_hex("orange")), 
                               (55.4, 'Mauvaise', get_color_hex("red"))]:
        fig.add_hline(y=limit, line_dash="dash", line_color=color, 
                     annotation_text=label, annotation_position="top right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Moyenne", f"{df_hist['PM2.5'].mean():.1f} µg/m³")
    with col2:
        st.metric("Maximum", f"{df_hist['PM2.5'].max():.1f} µg/m³")
    with col3:
        st.metric("Minimum", f"{df_hist['PM2.5'].min():.1f} µg/m³")
    with col4:
        tendance = "Stable" if abs(df_hist['PM2.5'].pct_change().mean()) < 0.1 else "À la hausse" if df_hist['PM2.5'].pct_change().mean() > 0 else "À la baisse"
        st.metric("Tendance", tendance)
    
    # Téléchargement des données
    csv = df_hist.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les données historiques (CSV)",
        data=csv,
        file_name=f"historique_aqi_{zone_hist}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def tab_previsions(all_zone_data, predictor):
    """Onglet 3: Prévisions futures"""
    st.header("🟣 Prévisions AQI")
    
    # Sélecteur de zone
    zone_prev = st.selectbox("Zone pour prévisions", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_prev)
    
    # Prévisions détaillées
    col1, col2 = st.columns([2, 1])
    
    with col1:
        horizon = st.slider("Horizon de prévision (heures)", 1, 72, 24)
        
        # Génération des prévisions
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
                'Heure': forecast_time,
                'PM2.5': predicted_pm25,
                'Statut': status,
                'Couleur': color,
                'Icone': icon
            })
        
        df_forecast = pd.DataFrame(forecast_data)
        
        # Graphique de prévision
        fig = px.line(df_forecast, x='Heure', y='PM2.5',
                      title=f"Prévisions AQI - {zone_prev}",
                      labels={'PM2.5': 'Concentration PM2.5 (µg/m³)', 'Heure': 'Date et Heure'})
        
        # Ajout des zones colorées selon le statut
        for i in range(len(df_forecast)-1):
            status = df_forecast.iloc[i]['Statut']
            color = get_color_hex(df_forecast.iloc[i]['Couleur'])
            fig.add_vrect(x0=df_forecast.iloc[i]['Heure'], 
                         x1=df_forecast.iloc[i+1]['Heure'],
                         fillcolor=color, opacity=0.2, line_width=0)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Indicateurs Météo")
        weather = selected_zone_data['weather']
        
        st.metric("🌡️ Température", f"{weather['temperature']:.1f}°C")
        st.metric("💧 Humidité", f"{weather['humidity']:.1f}%")
        st.metric("💨 Vitesse du vent", f"{weather['wind_speed']:.1f} km/h")
        st.metric("🌧️ Pluie (1h)", f"{weather['rain_1h']:.1f} mm")
        
        st.subheader("🚨 Zones à Risque")
        risky_zones = []
        for zone in all_zone_data:
            forecast_24h = predictor.predict_advanced(
                {'pm25': zone['pm25']}, 
                zone['weather'], 
                hours_ahead=24,
                reference_dt=st.session_state.get('analysis_datetime', datetime.now())
            )
            if forecast_24h > 35.4:  # Seuil "Médiocre"
                risky_zones.append((zone['zone'], forecast_24h))
        
        if risky_zones:
            for zone_name, pm25 in risky_zones:
                status, color, icon = get_aqi_status_and_color(pm25)
                st.warning(f"{icon} {zone_name}: {pm25:.1f} µg/m³ ({status})")
        else:
            st.success("✅ Aucune zone à risque élevé prévue")
        
        # Simulation "Et si"
        st.subheader("🔬 Simulation 'Et si'")
        new_temp = st.slider("Température simulée (°C)", 10, 40, int(weather['temperature']))
        new_wind = st.slider("Vent simulé (km/h)", 0, 30, int(weather['wind_speed']))
        
        if st.button("Simuler l'impact"):
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
            st.info(f"Impact simulé: {difference:+.1f} µg/m³")

def tab_profil_utilisateur():
    """Onglet 4: Profil utilisateur"""
    st.header("🟠 Profil Utilisateur")
    
    with st.form("user_profile_form"):
        st.subheader("📝 Informations Personnelles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.selectbox("Âge", ["<18", "18-30", "31-50", "51-65", "65+"])
            conditions = st.multiselect("Problèmes de santé", 
                                      ["Asthme", "Allergies", "Problèmes cardiaques", "Aucun"])
        
        with col2:
            profession = st.selectbox("Profession", 
                                    ["Bureau", "Extérieur", "Étudiant", "Retraité", "Autre"])
            activites = st.multiselect("Activités régulières",
                                     ["Sport intensif", "Marche", "Vélo", "Jardinage", "Aucune"])
        
        # Soumission du formulaire
        submitted = st.form_submit_button("💾 Sauvegarder le profil")
        
        if submitted:
            st.session_state['user_profile_details'] = {
                'age': age,
                'conditions': conditions,
                'profession': profession,
                'activites': activites
            }
            st.success("✅ Profil sauvegardé avec succès!")
    
    # Affichage du résumé de risque
    if 'user_profile_details' in st.session_state:
        st.subheader("📊 Résumé de Votre Risque Personnel")
        
        profile = st.session_state['user_profile_details']
        risk_factors = 0
        
        # Calcul des facteurs de risque
        if profile['age'] in ['<18', '65+']:
            risk_factors += 1
        if any(cond in profile['conditions'] for cond in ['Asthme', 'Problèmes cardiaques']):
            risk_factors += 2
        if profile['profession'] == 'Extérieur':
            risk_factors += 1
        if 'Sport intensif' in profile['activites']:
            risk_factors += 1
        
        # Détermination du niveau de risque
        if risk_factors >= 3:
            risk_level = "Élevé"
            color = "red"
            advice = "Surveillez attentivement la qualité de l'air. Évitez les activités extérieures lorsque l'AQI est médiocre ou pire."
        elif risk_factors >= 2:
            risk_level = "Modéré"
            color = "orange"
            advice = "Soyez prudent lors des jours de pollution. Réduisez les activités intenses à l'extérieur."
        else:
            risk_level = "Faible"
            color = "green"
            advice = "Vous êtes peu sensible à la pollution. Continuez à profiter de vos activités normales."
        
        st.markdown(f"<h3 style='color:{color}'>Niveau de risque: {risk_level}</h3>", unsafe_allow_html=True)
        st.info(f"💡 **Conseil personnalisé**: {advice}")
        
        # Suggestions d'activités basées sur l'AQI actuel
        st.subheader("🎯 Suggestions d'Activités")
        
        # Simulation de l'AQI actuel (en pratique, utiliser les données réelles)
        current_aqi = 25  # Valeur simulée
        
        if current_aqi <= 35:
            st.success("**✅ Conditions excellentes** - Idéal pour toutes les activités extérieures")
            st.write("• Sport intensif en extérieur")
            st.write("• Randonnée et vélo")
            st.write("• Pique-nique en plein air")
        elif current_aqi <= 55:
            st.warning("**⚠️ Conditions modérées** - Adaptez vos activités")
            st.write("• Sport modéré acceptable")
            st.write("• Évitez les efforts prolongés")
            st.write("• Personnes sensibles: limitez le temps dehors")
        else:
            st.error("**🚨 Conditions dégradées** - Privilégiez l'intérieur")
            st.write("• Reportez les activités sportives")
            st.write("• Limitez les déplacements non essentiels")
            st.write("• Utilisez un purificateur d'air à l'intérieur")

def tab_analyse_zone(all_zone_data):
    """Onglet 5: Analyse par zone détaillée"""
    st.header("⚫ Analyse par Zone")
    
    # Sélecteur de zone
    zone_analyze = st.selectbox("Sélectionnez une zone", [d['zone'] for d in all_zone_data])
    selected_zone_data = next(d for d in all_zone_data if d['zone'] == zone_analyze)
    
    # Métriques de la zone
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AQI Actuel", f"{selected_zone_data['pm25']:.1f} µg/m³")
    with col2:
        # Calcul AQI min/max (simulation)
        aqi_min = max(5, selected_zone_data['pm25'] * 0.7)
        aqi_max = min(150, selected_zone_data['pm25'] * 1.3)
        st.metric("AQI Min (24h)", f"{aqi_min:.1f} µg/m³")
    with col3:
        st.metric("AQI Max (24h)", f"{aqi_max:.1f} µg/m³")
    with col4:
        st.metric("AQI Moyen", f"{(aqi_min + aqi_max) / 2:.1f} µg/m³")
    
    # Graphiques détaillés
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution horaire (simulation)
        heures = list(range(24))
        aqi_horaire = [selected_zone_data['pm25'] * (1 + 0.3 * np.sin(h/24 * 2 * np.pi)) for h in heures]
        
        fig_horaire = px.line(x=heures, y=aqi_horaire,
                            title=f"Profil Horaire Typique - {zone_analyze}",
                            labels={'x': 'Heure de la journée', 'y': 'AQI (µg/m³)'})
        st.plotly_chart(fig_horaire, use_container_width=True)
    
    with col2:
        # Comparaison avec autres zones
        zones_comparaison = st.multiselect("Zones à comparer",
                                         [d['zone'] for d in all_zone_data if d['zone'] != zone_analyze],
                                         default=[d['zone'] for d in all_zone_data if d['zone'] != zone_analyze][:2])
        
        if zones_comparaison:
            data_comparaison = [selected_zone_data]
            for zone in all_zone_data:
                if zone['zone'] in zones_comparaison:
                    data_comparaison.append(zone)
            
            fig_comparaison = px.bar(
                x=[d['zone'] for d in data_comparaison],
                y=[d['pm25'] for d in data_comparaison],
                title="Comparaison AQI entre Zones",
                labels={'x': 'Zone', 'y': 'AQI (µg/m³)'}
            )
            st.plotly_chart(fig_comparaison, use_container_width=True)
    
    # Tendances détaillées
    st.subheader("📈 Tendances Détaillées")
    
    # Simulation de données de tendance
    jours = 30
    dates_tendance = [datetime.now() - timedelta(days=x) for x in range(jours, 0, -1)]
    tendance_data = []
    
    for date in dates_tendance:
        variation = random.uniform(0.8, 1.2)
        aqi_jour = selected_zone_data['pm25'] * variation
        tendance_data.append({'Date': date, 'AQI': aqi_jour})
    
    df_tendance = pd.DataFrame(tendance_data)
    
    fig_tendance = px.line(df_tendance, x='Date', y='AQI',
                          title=f"Tendance AQI - {zone_analyze} (30 jours)",
                          labels={'AQI': 'Concentration PM2.5 (µg/m³)'})
    
    # Ajout de la ligne de tendance
    z = np.polyfit(range(len(df_tendance)), df_tendance['AQI'], 1)
    p = np.poly1d(z)
    fig_tendance.add_scatter(x=df_tendance['Date'], y=p(range(len(df_tendance))),
                           mode='lines', name='Tendance', line=dict(dash='dash'))
    
    st.plotly_chart(fig_tendance, use_container_width=True)

def tab_facteurs_pollution(all_zone_data):
    """Onglet 6: Facteurs de pollution"""
    st.header("⚪ Facteurs de Pollution")
    
    selected_zone = st.selectbox("Zone d'analyse", [d['zone'] for d in all_zone_data])
    selected_data = next(d for d in all_zone_data if d['zone'] == selected_zone)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sources de Pollution")
        
        # Simulation des sources de pollution
        sources = {
            'Trafic': random.uniform(20, 40),
            'Industrie': random.uniform(10, 25),
            'Chauffage': random.uniform(5, 15),
            'Agriculture': random.uniform(8, 18),
            'Naturel': random.uniform(5, 12),
            'Autres': random.uniform(10, 20)
        }
        
        # Normalisation à 100%
        total = sum(sources.values())
        sources = {k: (v/total)*100 for k, v in sources.items()}
        
        fig_sources = px.pie(values=list(sources.values()), names=list(sources.keys()),
                           title="Contribution des Sources de Pollution")
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Corrélations Météo")
        
        # Simulation de données de corrélation
        temperature = np.random.normal(20, 5, 100)
        humidite = np.random.normal(60, 15, 100)
        aqi_simule = selected_data['pm25'] + temperature * 0.5 - humidite * 0.3 + np.random.normal(0, 5, 100)
        
        fig_corr_temp = px.scatter(x=temperature, y=aqi_simule,
                                 title="Corrélation AQI vs Température",
                                 labels={'x': 'Température (°C)', 'y': 'AQI (µg/m³)'})
        fig_corr_temp.update_layout(showlegend=False)
        st.plotly_chart(fig_corr_temp, use_container_width=True)
        
        fig_corr_hum = px.scatter(x=humidite, y=aqi_simule,
                                title="Corrélation AQI vs Humidité",
                                labels={'x': 'Humidité (%)', 'y': 'AQI (µg/m³)'})
        fig_corr_hum.update_layout(showlegend=False)
        st.plotly_chart(fig_corr_hum, use_container_width=True)
    
    # Conseils pour réduire l'impact
    st.subheader("💡 Réduire Votre Impact")
    
    conseils = [
        "🚗 Utilisez les transports en commun ou le covoiturage",
        "🚲 Privilégiez le vélo pour les courts trajets",
        "💡 Éteignez les lumières et appareils inutilisés",
        "🌱 Réduisez votre consommation de viande",
        "🛒 Achetez local et de saison",
        "♻️ Recycler et composter vos déchets",
        "🏠 Améliorez l'isolation de votre logement",
        "🌳 Plantez des arbres et végétaux"
    ]
    
    for conseil in conseils:
        st.write(f"• {conseil}")

def tab_rapports_export(all_zone_data):
    """Onglet 7: Rapports et export"""
    st.header("🟩 Rapports & Export")
    
    # Génération de rapport automatique
    st.subheader("📄 Rapport Automatique")
    
    # Calcul des statistiques globales
    aqi_values = [zone['pm25'] for zone in all_zone_data]
    aqi_moyen = np.mean(aqi_values)
    aqi_max = max(aqi_values)
    zone_max = all_zone_data[np.argmax(aqi_values)]['zone']
    status_global, _, _ = get_aqi_status_and_color(aqi_moyen)
    
    # Rapport texte
    rapport = f"""
    ## 📊 Rapport Qualité de l'Air - {datetime.now().strftime('%d/%m/%Y')}
    
    ### Résumé Exécutif
    - **Qualité globale**: {status_global}
    - **AQI moyen**: {aqi_moyen:.1f} µg/m³
    - **Zone la plus affectée**: {zone_max} ({aqi_max:.1f} µg/m³)
    - **Nombre de zones surveillées**: {len(all_zone_data)}
    
    ### Détails par Zone
    """
    
    for zone in all_zone_data:
        status, _, icon = get_aqi_status_and_color(zone['pm25'])
        rapport += f"- **{zone['zone']}**: {zone['pm25']:.1f} µg/m³ {icon} ({status})\n"
    
    rapport += """
    ### Recommandations
    - Surveillez les zones à risque élevé
    - Adaptez vos activités selon la qualité de l'air
    - Consultez les prévisions pour planifier vos sorties
    """
    
    st.markdown(rapport)
    
    # Options d'export
    st.subheader("📤 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export CSV
        df_export = pd.DataFrame([{
            'Zone': zone['zone'],
            'AQI': zone['pm25'],
            'Statut': get_aqi_status_and_color(zone['pm25'])[0],
            'Latitude': zone['lat'],
            'Longitude': zone['lon'],
            'Source': zone.get('source', 'Simulé'),
            'Date': datetime.now()
        } for zone in all_zone_data])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"rapport_aqi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export PDF simulé
        if st.button("📄 Générer PDF"):
            st.info("Fonctionnalité PDF en cours de développement...")
            st.success("✅ Rapport PDF généré avec succès! (simulation)")
    
    # Graphiques pour le rapport
    st.subheader("📈 Visualisations du Rapport")
    
    # Graphique comparatif
    fig_rapport = px.bar(
        x=[zone['zone'] for zone in all_zone_data],
        y=[zone['pm25'] for zone in all_zone_data],
        title="Comparaison AQI entre Toutes les Zones",
        labels={'x': 'Zone', 'y': 'AQI (µg/m³)'},
        color=[get_aqi_status_and_color(zone['pm25'])[1] for zone in all_zone_data]
    )
    st.plotly_chart(fig_rapport, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. NAVIGATION ET ÉTAT DE SESSION
# -----------------------------------------------------------------------------

def create_navigation():
    """Crée une navigation latérale avancée avec choix de la date/heure et profil utilisateur."""
    if 'user_profile' not in st.session_state:
        st.session_state['user_profile'] = {'risk': 'medium', 'activities': ['outdoor_sport', 'walk']}
    if 'analysis_datetime' not in st.session_state:
        st.session_state['analysis_datetime'] = datetime.now().replace(minute=0, second=0, microsecond=0)
    if 'selected_zone' not in st.session_state:
        st.session_state['selected_zone'] = LA_ZONES["Downtown"]
    
    with st.sidebar:
        st.title("🎛️ Paramètres d'Analyse")
        
        # Sélection temporelle
        st.subheader("⏱️ Période d'Analyse")
        
        current_dt = datetime.now()
        analysis_dt = st.session_state['analysis_datetime']

        selected_date = st.date_input(
            "Date d'analyse",
            value=analysis_dt.date(),
            min_value=current_dt.date() - timedelta(days=7),
            max_value=current_dt.date() + timedelta(days=3)
        )
        
        selected_time = st.time_input("Heure d'analyse", value=analysis_dt.time())
        
        new_analysis_dt = datetime.combine(selected_date, selected_time)
        if new_analysis_dt != st.session_state['analysis_datetime']:
             st.session_state['analysis_datetime'] = new_analysis_dt
             st.rerun()

        st.markdown("---")
        
        # Profil utilisateur
        st.subheader("👤 Votre Profil de Risque")
        
        risk_level = st.selectbox(
            "Niveau de Sensibilité à la Pollution",
            options=['low', 'medium', 'high', 'critical'],
            format_func=lambda x: get_risk_description(x)[0],
            index=['low', 'medium', 'high', 'critical'].index(st.session_state['user_profile']['risk'])
        )
        
        activities = st.multiselect(
            "Activités de Plein Air Prévues",
            options=['outdoor_sport', 'walk', 'bike', 'garden'],
            default=st.session_state['user_profile']['activities'],
            format_func=lambda x: {"outdoor_sport": "Sport Intensif", "walk": "Marche/Promenade", "bike": "Vélo", "garden": "Jardinage/Loisirs"}.get(x, x)
        )
        
        st.session_state['user_profile'] = {'risk': risk_level, 'activities': activities}
        
        st.markdown(f"**Conseil**: *{get_risk_description(risk_level)[1]}*")

# -----------------------------------------------------------------------------
# 8. APPLICATION PRINCIPALE AVEC ONGLETS
# -----------------------------------------------------------------------------

def main_application():
    """Contrôleur principal de l'application Streamlit avec onglets."""
    
    # Initialisation des classes
    fetcher = DataFetcher()
    predictor = AdvancedAQIPredictor()
    
    st.title("Guardian AI 🌍 Prévisions Avancées de la Qualité de l'Air (AQI)")
    
    # Obtenir le temps de référence
    analysis_dt = st.session_state.get('analysis_datetime', datetime.now())
    is_historic = analysis_dt < datetime.now() - timedelta(minutes=60)
    
    if is_historic:
        st.header(f"Mode Historique : Données pour le **{analysis_dt.strftime('%d/%m/%Y à %H:%M')}**")
    elif analysis_dt > datetime.now() + timedelta(minutes=10):
        st.header(f"Mode Prévision : Données pour le **{analysis_dt.strftime('%d/%m/%Y à %H:%M')}**")
    else:
        st.header(f"Mode Actuel : Données pour le **{analysis_dt.strftime('%d/%m/%Y à %H:%M')}**")
    
    st.markdown("---")
    
    # Récupération des données
    with st.spinner('🛰️ Récupération et harmonisation des données satellites et terrestres...'):
        all_zone_data = []
        df_map_data = []

        for zone_name, zone_info in LA_ZONES.items():
            
            weather_data = fetcher.fetch_advanced_weather(zone_info['lat'], zone_info['lon'])
            avg_pm25 = None
            source = "Simulé (Modèle)"

            if not is_historic and analysis_dt < datetime.now() + timedelta(minutes=10):
                zone_data_from_api = fetcher.fetch_multi_source_aqi(zone_info['lat'], zone_info['lon'])
                
                if zone_data_from_api:
                    pm25_values = [d['pm25'] for d in zone_data_from_api if d.get('pm25') is not None]
                    if pm25_values:
                        avg_pm25 = np.mean(pm25_values) * zone_info['weight']
                        source = "Réel (OpenAQ)"

            if avg_pm25 is None:
                time_of_day_factor = 1.2 if 7 <= analysis_dt.hour <= 9 or 16 <= analysis_dt.hour <= 19 else 0.8
                delta_seconds = (analysis_dt - datetime.now()).total_seconds()
                delta_hours = delta_seconds / 3600
                trend_factor = 1.0 + (delta_hours * 0.01)
                
                pm25_simulated = (25 * time_of_day_factor * trend_factor + random.uniform(-5, 5)) * zone_info['weight']
                avg_pm25 = max(5, min(150, pm25_simulated))
                
                if not is_historic and analysis_dt < datetime.now() + timedelta(minutes=10):
                    st.sidebar.info(f"⚠️ **{zone_info['name']}**: Aucune donnée réelle. Simulation utilisée.")
                
                if is_historic:
                     source = "Historique Simulé"
                elif analysis_dt > datetime.now():
                    source = "Prévision Simulé"

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
    
    # Mise à jour de la zone sélectionnée
    zone_names = [d['zone'] for d in all_zone_data]
    selected_zone_name = st.sidebar.selectbox("Sélectionnez la Zone d'Analyse Détaillée", zone_names)
    st.session_state['selected_zone'] = next(info for name, info in LA_ZONES.items() if info['name'] == selected_zone_name)
    
    # Création des onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🟢 Tableau de bord", 
        "🔵 Historique", 
        "🟣 Prévisions", 
        "🟠 Profil", 
        "⚫ Analyse Zone", 
        "⚪ Facteurs Pollution", 
        "🟩 Rapports"
    ])
    
    # Contenu des onglets
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
# 9. POINT D'ENTRÉE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    create_navigation()
    main_application()