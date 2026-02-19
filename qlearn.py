from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
import networkx as nx

app = Flask(__name__)

# --- 1. DEFINICIÓN DE ESTADOS ---
def_de_estados = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19
}
acciones = list(range(20))
indice_a_estado = {v: k for k, v in def_de_estados.items()}

# --- 2. CONEXIONES DEL GRAFO ---
conexiones = [
    ('A', 'B'), ('A', 'C'), ('A', 'K'), ('B', 'C'), ('C', 'G'),
    ('G', 'D'), ('G', 'H'), ('D', 'M'), ('D', 'O'), ('M', 'L'), 
    ('M', 'N'), ('M', 'O'), ('O', 'P'), ('P', 'Q'), ('Q', 'R'),
    ('R', 'S'), ('S', 'T'), ('T', 'K'), ('H', 'I'), ('I', 'J'),
    ('J', 'F'), ('F', 'E')
]

#FUNCION DE ENTRENAMIENTO
def entrenar_agente_estatico():
    gamma = 0.75
    alpha = 0.9
    
    # Inicialización de matriz R (Recompensas)
    R = np.zeros([20, 20])
    for u, v in conexiones:
        R[def_de_estados[u], def_de_estados[v]] = 1
        R[def_de_estados[v], def_de_estados[u]] = 1
    
    # UBICACIÓN OBJETIVO FIJA
    destino_final = 'G'
    idx_meta = def_de_estados[destino_final]
    R[idx_meta, idx_meta] = 1000  # Recompensa alta para fijar el destino
    
    # Inicialización de Valores Q (Original)
    Q = np.array(np.zeros([20, 20]))
    
    # Bucle de Entrenamiento
    for i in range(3000): # Aumentado a 3000 por ser 20 nodos
        estado_actual = np.random.randint(0, 20)
        accion_realizable = []
        for j in range(20):
            if R[estado_actual, j] > 0:
                accion_realizable.append(j)
        
        if not accion_realizable: continue
            
        accion_actual = np.random.choice(accion_realizable)
        estado_siguiente = accion_actual
        
        # --- ECUACIÓN DE BELLMAN / DIFERENCIA TEMPORAL (ORIGINAL) ---
        Ri = R[estado_actual, accion_actual]
        TD = Ri + gamma * Q[estado_siguiente, np.argmax(Q[estado_siguiente,])] - Q[estado_actual, accion_actual]
        Q[estado_actual, accion_actual] = Q[estado_actual, accion_actual] + alpha * TD
        
    return Q, destino_final

# Ejecutar entrenamiento al iniciar el servidor
Q_VALORES, META_FIJA = entrenar_agente_estatico()

# --- 4. INTERFAZ WEB ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Q-Learning Transporte - 20 Nodos</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 30px; background-color: #f0f2f5; }
        .container { display: flex; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { font-size: 10px; border-collapse: collapse; }
        td, th { border: 1px solid #ddd; padding: 3px; text-align: center; }
        .highlight { font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>🤖 Agente Entrenado para llegar a: <span style="color:red">{{ meta }}</span></h1>
    <div class="container">
        <div class="panel" style="width: 30%;">
            <h3>Calcular Ruta</h3>
            <label>Punto de Partida:</label>
            <select id="start" style="width:100%; padding:10px;">
                {% for nodo in nodos %}
                <option value="{{ nodo }}">{{ nodo }}</option>
                {% endfor %}
            </select>
            <button onclick="update()" style="width:100%; margin-top:10px; padding:10px; cursor:pointer;">Ver Ruta Óptima</button>
            <p id="route-text" class="highlight"></p>
            
            <h3>Matriz Q (Conocimiento)</h3>
            <div style="height: 300px; overflow: auto;">{{ q_table | safe }}</div>
        </div>
        <div class="panel" id="graph" style="width: 70%; height: 600px;"></div>
    </div>

    <script>
        function update() {
            let start = document.getElementById('start').value;
            fetch('/route?start=' + start).then(r => r.json()).then(data => {
                document.getElementById('route-text').innerText = "Ruta: " + data.path.join(" ➔ ");
                Plotly.newPlot('graph', data.plot.data, data.plot.layout);
            });
        }
        update();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    df_q = pd.DataFrame(Q_VALORES.astype(int), index=def_de_estados.keys(), columns=def_de_estados.keys())
    return render_template_string(HTML_TEMPLATE, 
                               meta=META_FIJA, 
                               nodos=sorted(def_de_estados.keys()), 
                               q_table=df_q.to_html())

@app.route('/route')
def get_route():
    start_node = request.args.get('start', 'A')
    
    # Lógica para extraer la ruta de la Matriz Q
    path = [start_node]
    actual = start_node
    for _ in range(20):
        if actual == META_FIJA: break
        idx = def_de_estados[actual]
        proximo_idx = np.argmax(Q_VALORES[idx])
        actual = indice_a_estado[proximo_idx]
        path.append(actual)

    # Preparar el grafo para Plotly
    G = nx.Graph()
    G.add_edges_from(conexiones)
    pos = nx.kamada_kawai_layout(G)
    
    edge_x, edge_y = [], []
    for u, v in conexiones:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    plot_data = [
        {"x": edge_x, "y": edge_y, "mode": "lines", "line": {"color": "#888", "width": 1}},
        {"x": [pos[n][0] for n in G.nodes()], "y": [pos[n][1] for n in G.nodes()], 
         "mode": "markers+text", "text": list(G.nodes()), "marker": {"size": 30, "color": "white", "line": {"width": 2}}},
        {"x": [pos[n][0] for n in path], "y": [pos[n][1] for n in path], 
         "mode": "lines+markers", "line": {"color": "red", "width": 5}, "marker": {"size": 35, "color": "orange"}}
    ]

    return jsonify({"path": path, "plot": {"data": plot_data, "layout": {"margin": {"l":20,"r":20,"t":20,"b":20}}}})

if __name__ == '__main__':
    app.run(debug=True)