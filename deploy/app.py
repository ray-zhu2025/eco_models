from flask import Flask, request, jsonify
import os
import subprocess
from datetime import datetime

app = Flask(__name__)

@app.route('/run_wrf', methods=['POST'])
def run_wrf():
    try:
        # 获取请求参数
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # 运行WRF模型
        cmd = f"mpirun -np {os.getenv('WRF_NUM_PROCESSORS', '4')} ./wrf.exe"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return jsonify({
                'status': 'success',
                'message': 'WRF model run completed successfully',
                'output': stdout.decode()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'WRF model run failed',
                'error': stderr.decode()
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 