
import sys
import os

# Agregar directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.server import app
from app.config import FLASK_CONFIG, ensure_directories

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 DEEP KRIGING - SISTEMA DE MONITOREO DE SEQUÍAS")
    print("="*60)
    print(f"\n📍 Región: Puno, Perú")
    print(f"🌐 URL: http://localhost:{FLASK_CONFIG['port']}")
    print(f"\n💡 Presiona Ctrl+C para detener")
    print("="*60 + "\n")
    
    # Verificar directorios
    ensure_directories()
    
    # Ejecutar Flask
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )
