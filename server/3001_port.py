from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"message": "人群生成的结果为12345"}
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"message": "人群生成的结果为12345"}
        self.wfile.write(json.dumps(response).encode())

httpd = HTTPServer(('localhost', 3001), SimpleHTTPRequestHandler)
httpd.serve_forever()
