# app.py (Backend con Firebase, Gemini AI, y Lógica de Negocio)

# --- Imports Nativos y de Flask ---
import os
import io
import math
import traceback
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# --- Imports de Librerías de Terceros ---
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from python_tsp.heuristics import solve_tsp_simulated_annealing

# --- Imports para Reportes ---
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# --- Imports para Firebase y Google AI (Gemini) ---
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# ==============================================================================
# --- CONFIGURACIÓN INICIAL Y CONEXIONES ---
# ==============================================================================

# --- 1. Inicialización de Firebase Admin SDK ---
# ¡ACCIÓN REQUERIDA! Asegúrate de que tu archivo de clave se llame 'serviceAccountKey.json'
# y esté en la misma carpeta que este script.
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print(">>> Conexión con Firestore establecida con éxito.")
except Exception as e:
    print(f"!!! ERROR: No se pudo conectar a Firebase. Verifica que 'serviceAccountKey.json' exista. Error: {e}")
    db = None

# --- 2. Configuración de Google AI (Gemini) ---
# ¡ACCIÓN REQUERIDA! Reemplaza "TU_API_KEY_DE_GEMINI" con tu clave real.
# Obtenla desde Google AI Studio: https://aistudio.google.com/app/apikey
try:
    GEMINI_API_KEY = "AIzaSyCJqbnAEBjPdOkUA6rs0CMJ93vCFUV8aas" # PEGA TU CLAVE DE GEMINI AQUÍ
    if GEMINI_API_KEY == "TU_API_KEY_DE_GEMINI":
        print(">>> ADVERTENCIA: La API Key de Gemini no está configurada. Las funciones de IA no estarán disponibles.")
        gemini_model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print(">>> Modelo de Gemini cargado y listo.")
except Exception as e:
    print(f"!!! ERROR: No se pudo configurar el modelo de Gemini. Error: {e}")
    gemini_model = None

# --- 3. Configuración de la App Flask ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- Constantes y Parámetros del Negocio ---
DEPOT_ID = "depot"
# ... (El resto de las constantes se mantienen igual)

# ==============================================================================
# --- LÓGICA DE NEGOCIO Y FUNCIONES AUXILIARES ---
# (Sin cambios en esta sección, se omite por brevedad)
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    if not all(isinstance(x, (float, int)) for x in [lat1, lon1, lat2, lon2]): return 0.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    return 6371000 * 2 * math.asin(min(1.0, math.sqrt(a)))

def _parse_csv(file_stream):
    try:
        df = pd.read_csv(file_stream, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        lat_col = next((c for c in df.columns if c in ['lat', 'latitude', 'latitud']), None)
        lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude', 'longitud']), None)
        if not lat_col or not lon_col: raise ValueError("Columnas de lat/lon no encontradas.")
        pas_col, col_col, nom_col = next((c for c in df.columns if c in ['pasajeros']), None), next((c for c in df.columns if c in ['color']), None), next((c for c in df.columns if c in ['nombre']), None)
        paradas = []
        for _, r in df.iterrows():
            try:
                paradas.append({
                    'lat': float(r[lat_col]), 'lon': float(r[lon_col]),
                    'pasajeros': int(r[pas_col]) if pas_col and pd.notna(r[pas_col]) else 1,
                    'color': str(r[col_col]).lower() if col_col and pd.notna(r[col_col]) else 'rojo',
                    'nombre': str(r[nom_col]) if nom_col and pd.notna(r[nom_col]) else 'Parada sin nombre'
                })
            except (ValueError, TypeError): continue
        return paradas
    except Exception as e:
        raise ValueError(f"Error procesando CSV: {e}")

# --- ¡NUEVA FUNCIÓN! Generador de Insights con Gemini AI ---
def generar_resumen_ia(ruta_data):
    if not gemini_model:
        return "El servicio de IA no está disponible."
    
    try:
        paradas_nombres = [p['nombre'] for p in ruta_data['paradas_info']]
        prompt = f"""
        Eres "Rout-IA", un asistente de logística experto para la ciudad de Guadalajara de Buga, Colombia.
        Analiza los datos de la siguiente ruta y genera un resumen conciso (máximo 4 líneas), útil y amigable para el conductor.

        Datos de la ruta:
        - Vehículo ID: {ruta_data['vehiculo_id']}
        - Total de Pasajeros: {ruta_data['total_pasajeros']} de {ruta_data['capacidad_sobrecupo']}
        - Utilización de Capacidad: {ruta_data['capacidad_utilizada_pct']:.1f}%
        - Distancia: {ruta_data['distancia_optima_m']/1000:.2f} km
        - Secuencia de Paradas: {' -> '.join(paradas_nombres)}

        Tu tarea:
        1. Escribe un titular breve y motivador.
        2. Menciona el número de paradas.
        3. Da un consejo práctico o resalta un dato clave (ej. "Ruta corta y eficiente" o "Vehículo casi lleno, buen trabajo").
        4. Finaliza con un mensaje de seguridad.
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"No se pudo generar el resumen de IA: {e}"

# ==============================================================================
# --- RUTAS DE LA API (ENDPOINTS) ---
# Ahora interactúan con Firestore
# ==============================================================================

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/paradas', methods=['GET'])
def get_paradas():
    if not db: return jsonify({"error": "Base de datos no disponible"}), 500
    try:
        paradas_ref = db.collection('paradas').stream()
        paradas = [doc.to_dict() for doc in paradas_ref]
        return jsonify(paradas), 200
    except Exception as e:
        return jsonify({"error": f"Error al leer de Firestore: {e}"}), 500

@app.route('/api/upload_paradas', methods=['POST'])
def upload_paradas_file():
    if not db: return jsonify({"error": "Base de datos no disponible"}), 500
    if 'paradasFile' not in request.files: return jsonify({"error": "No se encontró archivo"}), 400
    
    file = request.files['paradasFile']
    if not file or not file.filename.endswith('.csv'): return jsonify({"error": "Archivo no permitido, solo .csv"}), 400
    
    try:
        paradas_ref = db.collection('paradas')
        for doc in paradas_ref.stream():
            doc.reference.delete()

        paradas_cargadas = _parse_csv(io.TextIOWrapper(file.stream, encoding='utf-8'))
        
        batch = db.batch()
        for p_data in paradas_cargadas:
            doc_ref = db.collection('paradas').document()
            p_data['id'] = doc_ref.id
            p_data['timestamp'] = firestore.SERVER_TIMESTAMP
            batch.set(doc_ref, p_data)
        batch.commit()

        return jsonify({"message": f"{len(paradas_cargadas)} paradas cargadas con éxito."}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al procesar el archivo: {e}"}), 400

@app.route('/api/optimize', methods=['POST'])
def optimize_routes_full():
    if not db: return jsonify({"error": "Base de datos no disponible"}), 500

    data = request.get_json()
    if not data or 'vehiculos' not in data or 'depot' not in data:
        return jsonify({"error": "Datos de entrada incompletos"}), 400
    
    try:
        paradas_docs = db.collection('paradas').stream()
        paradas_list = [doc.to_dict() for doc in paradas_docs]
        if not paradas_list:
            return jsonify({"error": "No hay paradas en la base de datos para optimizar"}), 400

        paradas_df = pd.DataFrame(paradas_list)
        
        vehiculos_df = pd.DataFrame(data['vehiculos'])
        vehiculos_df['ID_Vehiculo'] = vehiculos_df.index + 1
        vehiculos_df['Sobrecupo'] = (vehiculos_df['capacidad'] * 1.05).astype(int)

        if paradas_df['pasajeros'].sum() > vehiculos_df['Sobrecupo'].sum():
            return jsonify({"error": "Capacidad de la flota insuficiente"}), 400

        n_clusters = min(len(vehiculos_df), len(paradas_df))
        if n_clusters == 0: return jsonify({"error": "No hay datos para clustering"}), 400

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        paradas_df['Cluster'] = kmeans.fit_predict(paradas_df[['lat', 'lon']])
        
        paradas_df_indexed = paradas_df.set_index('id')
        depot_coords = data['depot']
        depot_df = pd.DataFrame([{'lat': depot_coords['lat'], 'lon': depot_coords['lon'], 'pasajeros': 0}], index=[DEPOT_ID])
        paradas_df_con_depot = pd.concat([depot_df, paradas_df_indexed])
        
        rutas_asignadas = {v_id: [] for v_id in vehiculos_df['ID_Vehiculo']}
        vehiculos_para_asignacion = vehiculos_df['ID_Vehiculo'].tolist()[:n_clusters]
        for i in range(n_clusters):
            rutas_asignadas[vehiculos_para_asignacion[i]] = paradas_df[paradas_df['Cluster'] == i]['id'].tolist()
        
        resultados = []
        for vehiculo_id, paradas_ids in rutas_asignadas.items():
            if not paradas_ids: continue
            
            vehiculo_info = vehiculos_df[vehiculos_df['ID_Vehiculo'] == vehiculo_id].iloc[0]
            nodos_ids = [DEPOT_ID] + paradas_ids
            
            dist_matrix = np.array([[haversine(paradas_df_con_depot.loc[i]['lat'], paradas_df_con_depot.loc[i]['lon'], paradas_df_con_depot.loc[j]['lat'], paradas_df_con_depot.loc[j]['lon']) for j in nodos_ids] for i in nodos_ids])
            permutation, _ = solve_tsp_simulated_annealing(dist_matrix)
            secuencia_ids_ordenada = [nodos_ids[i] for i in permutation]
            
            start_idx = secuencia_ids_ordenada.index(DEPOT_ID)
            secuencia_final = secuencia_ids_ordenada[start_idx:] + secuencia_ids_ordenada[:start_idx]
            if secuencia_final[-1] != DEPOT_ID:
                secuencia_final.append(DEPOT_ID)
                
            distancia_total = 0
            for i in range(len(secuencia_final) - 1):
                distancia_total += haversine(
                    paradas_df_con_depot.loc[secuencia_final[i]]['lat'], paradas_df_con_depot.loc[secuencia_final[i]]['lon'],
                    paradas_df_con_depot.loc[secuencia_final[i+1]]['lat'], paradas_df_con_depot.loc[secuencia_final[i+1]]['lon']
                )

            ruta_resultado = {
                "vehiculo_id": int(vehiculo_id),
                "secuencia_paradas": [pid for pid in secuencia_final if pid != DEPOT_ID],
                "paradas_info": paradas_df_indexed.loc[paradas_ids].reset_index()[['id', 'nombre', 'pasajeros']].to_dict('records'),
                "total_pasajeros": int(paradas_df_indexed.loc[paradas_ids]['pasajeros'].sum()),
                "capacidad_sobrecupo": int(vehiculo_info['Sobrecupo']),
                "capacidad_utilizada_pct": (int(paradas_df_indexed.loc[paradas_ids]['pasajeros'].sum()) / int(vehiculo_info['Sobrecupo'])) * 100,
                "distancia_optima_m": distancia_total,
                "resumen_ia": ""
            }
            
            ruta_resultado['resumen_ia'] = generar_resumen_ia(ruta_resultado)
            resultados.append(ruta_resultado)

        return jsonify({ "rutas": resultados, "depot": depot_coords })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error interno en optimización: {str(e)}"}), 500

# ==============================================================================
# --- EJECUCIÓN DE LA APLICACIÓN ---
# ==============================================================================
if __name__ == '__main__':
    if not db:
        print("\n!!! LA APLICACIÓN SE EJECUTARÁ SIN CONEXIÓN A LA BASE DE DATOS !!!")
    app.run(host='0.0.0.0', port=5000, debug=True)
