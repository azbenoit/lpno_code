"""Static file server for local frontend dev that disables caching.

`python -m http.server` sends no cache headers, so browsers heuristically cache
`app.js`/`pvmap.js` and a normal reload keeps stale JS — which repeatedly looked
like "my fix didn't work". This sends `Cache-Control: no-store` on every response
so a plain reload always fetches the current files.

Usage: serve_frontend.py <directory> <bind> <port>
"""

from __future__ import annotations

import functools
import http.server
import sys


class _NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    directory, bind, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    handler = functools.partial(_NoCacheHandler, directory=directory)
    with http.server.ThreadingHTTPServer((bind, port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
