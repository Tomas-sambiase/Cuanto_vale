from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor Flask funcionando correctamente"

@app.route('/notificaciones', methods=['POST'])
def notificaciones():
    data = request.json
    print(f"Received notification: {data}")
    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
