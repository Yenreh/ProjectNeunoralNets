from flask import Flask
import sys
import argparse
import atexit

from helpers.utils import setup_experiment_environment
from controllers.gpu_config import configure_gpu
from controllers.routes import register_routes
from controllers.model_loader import cleanup_on_exit, load_available_models

# ============================================================================
# FLASK APPLICATION SETUP
# ============================================================================

# Configure GPU settings before loading models
configure_gpu()

# Setup experiment environment
setup_experiment_environment(42)

# Create Flask app
app = Flask(__name__)

# Register all routes
register_routes(app)


if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_on_exit)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the model demo Flask application")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the Flask app (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the Flask app (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Run in debug mode (default: True)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Iniciando aplicación Flask...")
    print("=" * 60)
    print(f"Modelos disponibles: {load_available_models()}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 60 + "\n")

    try:
        app.run(debug=args.debug, host=args.host, port=args.port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n Port {args.port} is already in use.")
            print(
                f"Try running with a different port: python run_demo.py --port {args.port + 1}"
            )
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("\nCerrando aplicación por interrupción del usuario...")
        cleanup_on_exit()
