import pytest
import requests
import requests_mock
from datetime import datetime
import P2pflWebServices

"""
Test the P2PFL web services.

Note: Not tested yet.
"""

@pytest.fixture
def web_services():
    return P2pflWebServices('https://p2pfl.com', 'test_api_key')

@pytest.fixture
def mock_api():
    with requests_mock.Mocker() as m:
        yield m

def test_register_node(web_services, mock_api, monkeypatch):
 # Mock datetime.now() to return a fixed date
    fixed_date = datetime(2023, 1, 1, 12, 0, 0)
    monkeypatch.setattr(datetime, 'now', lambda: fixed_date)

    mock_api.post('https://p2pfl.com/node', json={'node_id': 1})
    
    web_services.register_node('node1', False)
    
    assert web_services.node_id['node1'] == 1
    assert mock_api.call_count == 1
    
    last_request = mock_api.last_request
    assert last_request.json() == {
        'address': 'node1',
        'is_simulated': False,
        'creation_date': '2023-01-01 12:00:00'
    }
    assert last_request.headers['Content-Type'] == 'application/json'
    assert last_request.headers['x-api-key'] == 'test_api_key'

def test_register_node_error(web_services, mock_api):
    mock_api.post('https://p2pfl.com/node', status_code=500, text='Internal Server Error')
    
    with pytest.raises(requests.exceptions.RequestException) as exc_info:
        web_services.register_node('node1', False)
    
    assert '500' in str(exc_info.value)
    assert 'Internal Server Error' in str(exc_info.value)

def test_register_node_http_warning(capsys):
    P2pflWebServices('http://p2pfl.com', 'test_api_key')
    captured = capsys.readouterr()
    assert "Connection must be over https" in captured.out