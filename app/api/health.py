from flask import Blueprint, jsonify

bp = Blueprint('health', __name__)

@bp.route('/')
def health():
    return jsonify({'status': 'ok'})