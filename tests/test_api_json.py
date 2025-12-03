"""
API JSON safety tests (requires Flask test client)
"""
import json
import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


def test_valid_request(client):
    r = client.post('/api/chat', json={'message': 'hello'})
    assert r.status_code == 200
    data = r.get_json()
    assert data is not None
    assert 'answer' in data


def test_empty_body(client):
    r = client.post('/api/chat', data='')
    assert r.status_code == 400
    data = r.get_json()
    assert data is not None
    assert data.get('answer')


def test_malformed_json(client):
    r = client.post('/api/chat', data='{ bad json')
    assert r.status_code == 400
    data = r.get_json()
    assert data is not None
    assert data.get('answer')
