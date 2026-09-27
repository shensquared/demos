#!/usr/bin/env python3
"""Serve this folder's demos over HTTP, with a portal page at /.

    python3 serve.py --lec 4                    6.390 lecture 4, from its local build
    python3 serve.py --build <folder>           any slides.com zip build
    python3 serve.py nll/a.html gd/gd3d.html    an explicit list, each as a full slide
    python3 serve.py                            no portal, / is a directory listing

Demos come from the deck's local zip build, never from the slides.com API. --lec
takes the build under ~/code/slides/390/<term>/_upstream/ whose deck title names
that lecture, the newest if several do. The portal lists each demo with its slide
number and the iframe block it fills on that slide, plus a slide view that places
the demo at that block inside a 1280x720 frame. The server listens on the LAN unless
--local is given, sends no-store so a reload shows the working tree, and refuses any
path with a dot segment such as .git. The build is read once at startup, so restart
after editing it.
"""
import argparse
import html
import html.parser
import http.server
import io
import json
import re
import socket
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILDS = Path.home() / "code/slides/390"  # builds sit at <term>/_upstream/<build>/
DEMO_HOST = "shenshen.mit.edu/demos/"
FULL = [0, 0, 1280, 720]


def deck_title(index):
    m = re.search(r"<title>(.*?)</title>", index.read_text(errors="ignore"), re.S | re.I)
    return html.unescape(m.group(1).strip()) if m else index.parent.name


def build_for_lecture(n):
    # Build folders carry no fixed name (lec4_with_before), so match on the deck
    # title the export writes into index.html: "... - Lecture 4 Linear Classification".
    builds = sorted(BUILDS.glob("*/_upstream/*/index.html"), key=lambda f: f.stat().st_mtime)
    matches = [f for f in builds if re.search(rf"\bLecture\s+{n}\b", deck_title(f), re.I)]
    if not matches:
        found = "\n".join(f"  {f.parent}  {deck_title(f)}" for f in builds) or "  none"
        sys.exit(f"no local build for lecture {n}. Builds found:\n{found}")
    return matches[-1].parent


def parse_box(style):
    v = dict(re.findall(r"(left|top|width|height):\s*(-?[\d.e+-]+)px", style))
    try:
        return [round(float(v[k])) for k in ("left", "top", "width", "height")]
    except (KeyError, ValueError):
        return FULL


class DeckIframes(html.parser.HTMLParser):
    """Collect (slide number, demo path, block box) for each iframe on this host.

    A slide is a leaf <section>. A section that turns out to hold sections is a
    vertical stack, so it hands its number to its first child.
    """

    def __init__(self):
        super().__init__()
        self.open = []  # [number, has_child] per open section
        self.count = 0
        self.box = FULL
        self.found = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section":
            if self.open and not self.open[-1][1]:
                self.open[-1][1] = True
                self.count -= 1
            self.count += 1
            self.open.append([self.count, False])
        elif tag == "div" and "sl-block" in (a.get("class") or "").split():
            self.box = parse_box(a.get("style") or "")
        elif tag == "iframe" and self.open:
            src = a.get("data-src") or a.get("src") or ""
            if DEMO_HOST in src:
                self.found.append((self.open[-1][0], src.split(DEMO_HOST, 1)[1], self.box))

    def handle_endtag(self, tag):
        if tag == "section" and self.open:
            self.open.pop()


def demo_title(path):
    f = ROOT / urllib.parse.urlsplit(path).path
    if f.is_dir():
        f = f / "index.html"
    try:
        m = re.search(r"<title>(.*?)</title>", f.read_text(errors="ignore"), re.S | re.I)
    except OSError:
        return f"{path} (not in this folder)"
    return html.unescape(m.group(1).strip()) if m else path


def portal_data(args):
    if args.build or args.lec:
        build = Path(args.build).expanduser() if args.build else build_for_lecture(args.lec)
        index = build / "index.html" if build.is_dir() else build
        parser = DeckIframes()
        parser.feed(index.read_text(errors="ignore"))
        demos = [{"slide": n, "path": p, "box": b} for n, p, b in parser.found]
        heading = deck_title(index)
        print(f"build: {index.parent}")
    else:
        demos = [{"slide": None, "path": p.removeprefix("./").lstrip("/"), "box": FULL} for p in args.paths]
        heading = f"{ROOT.name} demos"
    for d in demos:
        d["title"] = demo_title(d["path"])
    return {"heading": heading, "demos": demos}


class Handler(http.server.SimpleHTTPRequestHandler):
    portal = None  # bytes, or None to fall back to the directory listing

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        if any(part.startswith(".") for part in path.split("/") if part):
            self.send_error(404)
            return None
        if self.portal and path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.portal)))
            self.end_headers()
            return io.BytesIO(self.portal)
        return super().send_head()


def lan_ip():
    # Connecting a UDP socket sends nothing; it only picks the outbound interface.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="demo paths relative to this folder")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--lec", type=int, help=f"6.390 lecture number, matched against the builds under {BUILDS}")
    source.add_argument("--build", help="a zip build folder, or its index.html")
    ap.add_argument("--port", type=int, default=8390)
    ap.add_argument("--local", action="store_true", help="listen on 127.0.0.1 only")
    args = ap.parse_args()
    # Line-buffer so the URLs show up at once when output goes to a log, not a tty.
    sys.stdout.reconfigure(line_buffering=True)
    if args.paths and (args.lec or args.build):
        ap.error("give demo paths or a build, not both")

    if args.paths or args.lec or args.build:
        data = portal_data(args)
        payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        Handler.portal = PORTAL.replace("__DATA__", payload).encode()
        print(data["heading"])
        for d in data["demos"]:
            where = f"slide {d['slide']:>2}" if d["slide"] else "       "
            print(f"  {where}  {d['path']}  {d['title']}")

    host = "127.0.0.1" if args.local else "0.0.0.0"
    server = http.server.ThreadingHTTPServer((host, args.port), Handler)
    print(f"http://localhost:{args.port}/")
    ip = None if args.local else lan_ip()
    if ip:
        print(f"http://{ip}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PORTAL = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Demos</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #fafafa; color: #2b2b2b;
    }
    a { color: inherit; }
    a:focus-visible { outline: 2px solid #2b2b2b; outline-offset: 3px; }
    .mono { font-family: "SF Mono", Menlo, monospace; }

    main { max-width: 880px; margin: 0 auto; padding: 56px 24px; }
    h1 { font-size: 30px; font-weight: 600; letter-spacing: -0.01em; text-wrap: balance; }
    .sub { font-size: 18px; color: #5f6672; margin-top: 8px; }

    ol { list-style: none; margin-top: 36px; }
    li {
      display: grid; grid-template-columns: 88px 1fr auto;
      gap: 20px; align-items: baseline;
      padding: 20px 0; border-bottom: 1px solid #e2e2e2;
    }
    .slide { font-size: 16px; color: #5f6672; }
    .title { font-size: 22px; font-weight: 500; text-decoration: none; }
    .title:hover { text-decoration: underline; text-underline-offset: 4px; }
    .meta { font-size: 15px; color: #5f6672; margin-top: 6px; }
    .links { display: flex; gap: 18px; font-size: 16px; white-space: nowrap; }
    .links a { color: #3a7ebf; text-decoration: none; }
    .links a:hover { text-decoration: underline; text-underline-offset: 3px; }

    /* Slide view: the demo sits in a 1280x720 slide at its deck block position,
       scaled to the window the way reveal scales a deck. */
    #stage { display: none; position: fixed; inset: 0; background: #e6e6e6; }
    body.viewing main { display: none; }
    body.viewing #stage { display: block; }
    .bar {
      height: 52px; padding: 0 20px;
      display: flex; align-items: center; gap: 24px;
      font-size: 16px; color: #5f6672;
    }
    .bar a { text-decoration: none; }
    .bar a:hover { color: #2b2b2b; }
    .bar .now { color: #2b2b2b; font-weight: 500; }
    .bar .hint { margin-left: auto; }
    #slide {
      position: absolute; left: 50%; top: 52px;
      width: 1280px; height: 720px;
      background: #fff; box-shadow: 0 1px 4px rgba(0, 0, 0, .12);
      transform-origin: top center;
    }
    #slide iframe { position: absolute; border: 0; }

    @media (max-width: 640px) {
      li { grid-template-columns: 1fr; gap: 6px; }
    }
  </style>
</head>
<body>
  <main>
    <h1 id="heading"></h1>
    <p class="sub">Served from the working tree, so a reload shows the latest edit.</p>
    <ol id="list"></ol>
  </main>

  <div id="stage">
    <div class="bar">
      <a href="#">All demos</a>
      <a id="prev" href="#">Previous</a>
      <span class="now" id="now"></span>
      <a id="next" href="#">Next</a>
      <span class="hint">Esc for the list, [ and ] to step</span>
    </div>
    <div id="slide"></div>
  </div>

  <script>
    const DATA = __DATA__;
    const DEMOS = DATA.demos;

    const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const withEmbed = (p) => (/[?&]embed\b/.test(p) ? p : p + (p.includes('?') ? '&' : '?') + 'embed');
    const label = (d) => (d.slide ? `Slide ${d.slide}: ` : '') + d.title;

    document.title = DATA.heading;
    document.getElementById('heading').textContent = DATA.heading;
    document.getElementById('list').innerHTML = DEMOS.map((d, i) => {
      const [x, y, w, h] = d.box;
      const where = x === 0 && y === 0 && w === 1280 && h === 720 ? 'full slide' : `${w}×${h} inset at ${x}, ${y}`;
      return `<li>
        <span class="slide">${d.slide ? 'slide ' + d.slide : ''}</span>
        <div>
          <a class="title" href="/${esc(d.path)}">${esc(d.title)}</a>
          <div class="meta"><span class="mono">${esc(d.path)}</span>, ${where}</div>
        </div>
        <span class="links">
          <a href="#d${i + 1}">slide view</a>
          <a href="/${esc(withEmbed(d.path))}">embed</a>
        </span>
      </li>`;
    }).join('');

    const slideEl = document.getElementById('slide');
    let current = -1;

    function fit() {
      const s = Math.min(window.innerWidth / 1280, (window.innerHeight - 52) / 720);
      slideEl.style.transform = `translateX(-50%) scale(${s})`;
    }

    function show() {
      const m = location.hash.match(/^#d(\d+)$/);
      const i = m && DEMOS[Number(m[1]) - 1] ? Number(m[1]) - 1 : -1;
      document.body.classList.toggle('viewing', i >= 0);
      if (i < 0) { slideEl.innerHTML = ''; current = -1; return; }
      if (i === current) return;
      current = i;
      const d = DEMOS[i];
      const [x, y, w, h] = d.box;
      slideEl.innerHTML = `<iframe src="/${esc(d.path)}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px"></iframe>`;
      document.getElementById('now').textContent = label(d);
      document.getElementById('prev').href = '#d' + ((i + DEMOS.length - 1) % DEMOS.length + 1);
      document.getElementById('next').href = '#d' + ((i + 1) % DEMOS.length + 1);
      fit();
    }

    // Keys pressed inside a demo stay in its iframe, so [ and ] only step when the
    // bar or the grey margin has focus, and a slider never gets hijacked.
    document.addEventListener('keydown', (e) => {
      if (current < 0) return;
      if (e.key === 'Escape') location.hash = '';
      if (e.key === ']') location.hash = document.getElementById('next').getAttribute('href');
      if (e.key === '[') location.hash = document.getElementById('prev').getAttribute('href');
    });
    window.addEventListener('hashchange', show);
    window.addEventListener('resize', fit);
    show();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
