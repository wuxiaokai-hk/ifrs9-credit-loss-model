"""
mydbg.py - step through / record / replay only YOUR code, skip libraries (pandas, numpy, ...).

1) STEP (like pdb, but `s` never enters library code)
      python mydbg.py step my_model.py [script args...]
   or start from a point inside your code:   from mydbg import set_trace; set_trace()

2) RECORD (clickable HTML report + every call's real inputs/outputs saved to disk)
      python mydbg.py record my_model.py [script args...]
         -> trace.html         open in a browser
         -> trace_calls/       one .pkl per call of your functions + index.csv
      options (put them BEFORE the script name):
        --out FILE        report file                          (default trace.html)
        --root DIR        what counts as "your code"           (default: script's folder)
        --max-calls N     calls per function recorded+saved    (default 20; 0 = all)
        --max-lines N     code lines shown per call            (default 2000)
        --max-mb N        skip saving a call bigger than N MB  (default 200)
        --rows N          DataFrame preview rows in report     (default 5)

3) REPLAY a saved call (in Python / Jupyter / Spyder):
      from mydbg import load_call, list_calls
      list_calls()                          # table of all saved calls
      c = load_call(7)                      # or load_call("trace_calls/00007_add_features.pkl")
      c.inputs      c.globals      c.output      c.raised
      c.rerun()                             # run the function again on the saved inputs
      c.rerun(lags=5)                       # ... with some inputs changed
      c.check()                             # rerun and compare with the saved output
      c.debug()                             # step through it in pdb with the saved inputs

pdb commands:  n next | s step in | r run to return | c continue | l / ll list
               p x / pp x print | b file:line | u/d up/down | w where | q quit
"""
import os
import sys
import pdb
import csv
import dis
import html
import time
import types
import pickle
import reprlib
import datetime
import importlib
import importlib.util
from collections import Counter

_THIS = os.path.abspath(__file__)
_EXCLUDE = ("site-packages", "dist-packages", f"{os.sep}.venv{os.sep}",
            f"{os.sep}venv{os.sep}", f"{os.sep}env{os.sep}")
_CO_VARARGS, _CO_VARKW = 0x04, 0x08
_CO_GENLIKE = 0x20 | 0x80 | 0x200          # generator / coroutine / async generator


class _Filter:
    def __init__(self, root):
        self.root = os.path.abspath(root) + os.sep
        self._cache = {}

    def __call__(self, filename):
        r = self._cache.get(filename)
        if r is None:
            f = os.path.abspath(filename)
            r = (not filename.startswith("<")      # <frozen ...>, <string>, ...
                 and f.startswith(self.root) and f != _THIS
                 and not any(x in f for x in _EXCLUDE))
            self._cache[filename] = r
        return r


# ---------------------------------------------------------------- step mode
class ProjectPdb(pdb.Pdb):
    """pdb that only ever stops in files under `root`."""

    def __init__(self, root, **kw):
        super().__init__(**kw)
        self._mine = _Filter(root)

    def stop_here(self, frame):
        if not self._mine(frame.f_code.co_filename):
            return False
        return super().stop_here(frame)


def set_trace(root=None):
    frame = sys._getframe().f_back
    root = root or os.path.dirname(os.path.abspath(frame.f_code.co_filename))
    ProjectPdb(root).set_trace(frame)


# ------------------------------------------------------------ value display
def _short(v, maxlen=120):
    """One-line summary of any value (never prints a whole DataFrame)."""
    try:
        if isinstance(v, Unsaved):
            return repr(v)
        mod = type(v).__module__
        if mod.startswith("pandas"):
            if hasattr(v, "columns"):
                cols = list(v.columns)
                more = f" +{len(cols) - 8}" if len(cols) > 8 else ""
                return f"DataFrame{v.shape} cols={cols[:8]}{more}"
            if hasattr(v, "dtype") and hasattr(v, "shape"):
                return f"{type(v).__name__}{v.shape} dtype={v.dtype} head={v.head(3).tolist()}"
        if mod.startswith("numpy") and getattr(v, "shape", ()) != ():
            return f"ndarray{v.shape} dtype={v.dtype}"
        r = reprlib.Repr()
        r.maxstring = r.maxother = maxlen
        r.maxlist = r.maxdict = r.maxtuple = r.maxset = 8
        s = r.repr(v)
    except Exception as e:
        s = f"<repr failed: {type(e).__name__}>"
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def _value_html(v, rows):
    """Summary + (for tables/arrays) a click-to-open preview, captured NOW."""
    summary = html.escape(_short(v, 200))
    preview = None
    try:
        mod = type(v).__module__
        if mod.startswith("pandas") and hasattr(v, "head"):
            obj = v.to_frame() if not hasattr(v, "columns") else v
            obj = obj.iloc[:rows, :15]
            preview = obj.to_html(border=0, classes="df", max_cols=15)
            if hasattr(v, "dtypes") and hasattr(v, "columns"):
                dt = ", ".join(f"{c}: {t}" for c, t in list(v.dtypes.items())[:15])
                preview += f'<div class="dtypes">dtypes: {html.escape(dt)}</div>'
        elif mod.startswith("numpy") and getattr(v, "ndim", 0) >= 1:
            import numpy as np
            preview = "<pre>" + html.escape(np.array2string(v[:rows], threshold=60, max_line_width=110)) + "</pre>"
        elif isinstance(v, (dict, list, tuple, set)) and len(v) > 8:
            import pprint
            txt = pprint.pformat(v if not isinstance(v, dict) else dict(list(v.items())[:30]),
                                 width=100, compact=True)
            preview = "<pre>" + html.escape(txt[:3000]) + "</pre>"
    except Exception:
        preview = None
    if preview:
        return f'<details class="val"><summary><code>{summary}</code></summary>{preview}</details>'
    return f"<code>{summary}</code>"


# ---------------------------------------------------------- saving values
class Unsaved:
    """Placeholder for a value that could not be pickled (open file, DB connection, ...)."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"<not saved: {self.text}>"


def _dumps_dict(d):
    """Pickle a {name: value} dict NOW (snapshot); unpicklable items become Unsaved."""
    try:
        return pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        safe = {}
        for k, v in d.items():
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                safe[k] = v
            except Exception as e:
                safe[k] = Unsaved(f"{_short(v, 80)} ({type(e).__name__})")
        return pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)


_GLOBAL_CACHE = {}


def _global_names(code):
    """Names a function reads as globals (incl. inside its comprehensions/lambdas)."""
    r = _GLOBAL_CACHE.get(code)
    if r is None:
        names = set()
        stack = [code]
        while stack:
            c = stack.pop()
            for ins in dis.get_instructions(c):
                if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str):
                    names.add(ins.argval)
            stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
        r = _GLOBAL_CACHE[code] = tuple(sorted(names))
    return r


def _is_data(v):
    return not (isinstance(v, (types.ModuleType, type)) or callable(v))


# -------------------------------------------------------------- record mode
class _Node:
    __slots__ = ("name", "file", "lineno", "hidden", "args", "globs", "ret", "items",
                 "t0", "t1", "pending_exc", "raised", "snap", "last_line",
                 "nlines", "skipped_lines", "skipped_calls",
                 "cid", "parent_cid", "meta", "in_bytes", "pkl")

    def __init__(self, name, file, lineno, hidden):
        self.name, self.file, self.lineno, self.hidden = name, file, lineno, hidden
        self.args, self.globs, self.ret, self.items = [], [], None, []
        self.t0, self.t1 = time.perf_counter(), None
        self.pending_exc, self.raised = None, None
        self.snap, self.last_line = {}, None
        self.nlines, self.skipped_lines = 0, 0
        self.skipped_calls = Counter()
        self.cid = self.parent_cid = self.meta = self.in_bytes = self.pkl = None


def _interesting(k, v):
    return (not (k.startswith("__") and k.endswith("__"))
            and type(v).__name__ not in ("module", "function", "type", "builtin_function_or_method"))


class Recorder:
    def __init__(self, root, script, data_dir, max_calls=20, max_lines=2000, rows=5, max_mb=200):
        self.mine = _Filter(root)
        self.root = os.path.abspath(root)
        self.script = os.path.abspath(script)
        self.data_dir = os.path.abspath(data_dir)
        self.max_calls, self.max_lines, self.rows = max_calls, max_lines, rows
        self.max_bytes = max_mb * 1024 * 1024
        self.stack, self.roots = [], []
        self.counts = Counter()
        self.src = {}
        self.next_id = 1
        self.index = []

    # -- helpers
    def _line(self, fn, n):
        if fn not in self.src:
            try:
                with open(fn, encoding="utf-8", errors="replace") as f:
                    self.src[fn] = f.read().splitlines()
            except OSError:
                self.src[fn] = []
        L = self.src[fn]
        return L[n - 1].rstrip() if 0 < n <= len(L) else ""

    def _diff(self, node, frame):
        new = {k: _short(v) for k, v in frame.f_locals.items() if _interesting(k, v)}
        if node.last_line is not None:
            for k, v in new.items():
                if node.snap.get(k) != v:
                    node.last_line[3].append((k, v))
        node.snap = new

    def _capture_call(self, node, frame, parent):
        """Snapshot inputs (and globals the function reads) at call time."""
        co = frame.f_code
        n = (co.co_argcount + co.co_kwonlyargcount
             + bool(co.co_flags & _CO_VARARGS) + bool(co.co_flags & _CO_VARKW))
        names = co.co_varnames[:n]
        loc = frame.f_locals
        inputs = {k: loc[k] for k in names if k in loc}
        g = frame.f_globals
        globs = {k: g[k] for k in _global_names(co)
                 if k in g and not k.startswith("__") and _is_data(g[k])}
        node.args = [(k, _short(v, 40), _value_html(v, self.rows)) for k, v in inputs.items()]
        node.globs = [(k, _value_html(v, self.rows)) for k, v in globs.items()]
        node.cid = self.next_id
        self.next_id += 1
        node.parent_cid = parent.cid if parent else None
        if co.co_flags & _CO_GENLIKE:
            return                                    # generator resumes are not real calls
        node.meta = {
            "id": node.cid, "name": co.co_name, "qualname": getattr(co, "co_qualname", co.co_name),
            "module": g.get("__name__"), "file": os.path.abspath(co.co_filename),
            "line": co.co_firstlineno, "root": self.root, "script": self.script,
            "argcount": co.co_argcount, "posonly": co.co_posonlyargcount,
            "kwonly": co.co_kwonlyargcount,
            "varargs": bool(co.co_flags & _CO_VARARGS), "varkw": bool(co.co_flags & _CO_VARKW),
            "names": list(names), "parent": node.parent_cid,
        }
        node.in_bytes = (_dumps_dict(inputs), _dumps_dict(globs))

    def _save_call(self, node, ret):
        if node.meta is None:
            return
        status = "raised" if node.raised else "ok"
        out_b = None if node.raised else _dumps_dict({"output": ret})
        inp_b, glob_b = node.in_bytes
        node.in_bytes = None
        size = len(inp_b) + len(glob_b) + (len(out_b) if out_b else 0)
        meta = dict(node.meta, status=status, raised=node.raised,
                    ms=round((node.t1 - node.t0) * 1000, 3))
        safe = "".join(ch if ch.isalnum() or ch == "_" else "" for ch in node.name) or "fn"
        fname = f"{node.cid:05d}_{safe}.pkl"
        if size > self.max_bytes:
            status = f"not saved ({size / 1e6:.0f} MB > --max-mb)"
        else:
            with open(os.path.join(self.data_dir, fname), "wb") as f:
                pickle.dump({"meta": meta, "inputs": inp_b, "globals": glob_b, "output": out_b},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            node.pkl = os.path.join(self.data_dir, fname)
        self.index.append({"id": node.cid, "function": meta["qualname"],
                           "file": os.path.relpath(meta["file"], self.root), "line": meta["line"],
                           "parent": node.parent_cid or "", "status": status, "ms": meta["ms"],
                           "pkl": fname if node.pkl else ""})

    # -- tracing
    def global_trace(self, frame, event, arg):
        if event != "call" or not self.mine(frame.f_code.co_filename):
            return None
        code = frame.f_code
        name = code.co_name
        if name.startswith("<") and name not in ("<module>", "<lambda>"):
            return None                 # genexpr / comprehension: transparent
        if name != "<module>" and not (code.co_flags & 0x01):
            return None                 # class body (not a real function call)
        parent = self.stack[-1] if self.stack else None
        hidden = bool(parent and parent.hidden)
        if not hidden and name != "<module>":
            key = (code.co_filename, code.co_firstlineno, name)
            self.counts[key] += 1
            if self.max_calls and self.counts[key] > self.max_calls:
                hidden = True
                if parent:
                    parent.skipped_calls[name] += 1
        node = _Node(name, code.co_filename, code.co_firstlineno, hidden)
        if not hidden:
            if name != "<module>":
                self._capture_call(node, frame, parent)
            node.snap = {k: _short(v) for k, v in frame.f_locals.items() if _interesting(k, v)}
            (parent.items if parent else self.roots).append(("call", node))
        self.stack.append(node)
        node.t0 = time.perf_counter()          # don't count our own pickling time
        return self.local_trace

    def local_trace(self, frame, event, arg):
        node = self.stack[-1]
        if node.hidden:
            if event == "return":
                self.stack.pop()
            return self.local_trace
        if event == "line":
            node.pending_exc = None
            self._diff(node, frame)
            if node.nlines >= self.max_lines:
                node.skipped_lines += 1
                node.last_line = None
            else:
                item = ["line", frame.f_lineno, self._line(frame.f_code.co_filename, frame.f_lineno), []]
                node.items.append(item)
                node.last_line = item
                node.nlines += 1
        elif event == "exception":
            node.pending_exc = arg
            node.items.append(("exc", f"{arg[0].__name__}: {_short(arg[1], 200)}"))
        elif event == "return":
            node.t1 = time.perf_counter()
            self._diff(node, frame)
            if node.pending_exc is not None:
                node.raised = f"{node.pending_exc[0].__name__}: {_short(node.pending_exc[1], 200)}"
            elif node.name != "<module>":
                node.ret = (_short(arg, 40), _value_html(arg, self.rows))
            self._save_call(node, arg)
            self.stack.pop()
        return self.local_trace

    # -- outputs
    def write_index(self):
        with open(os.path.join(self.data_dir, "index.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "function", "file", "line", "parent", "status", "ms", "pkl"])
            w.writeheader()
            w.writerows(sorted(self.index, key=lambda r: r["id"]))

    def _rel(self, fn):
        return os.path.relpath(fn, self.root)

    def _render(self, n, out, depth):
        e = html.escape
        dur = ""
        if n.t1 is not None:
            ms = (n.t1 - n.t0) * 1000
            dur = f"{ms:.1f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"
        tag = f'<span class="cid">#{n.cid}</span> ' if n.cid else ""
        if n.name == "<module>":
            title = f'<span class="fn">{e(self._rel(n.file))}</span>'
        else:
            args = ", ".join(f'{e(k)}=<span class="a">{e(s)}</span>' for k, s, _ in n.args)
            if n.raised:
                res = f'<span class="raise">raised {e(n.raised.split(":")[0])}</span>'
            elif n.ret:
                res = f'<span class="ret">{e(n.ret[0])}</span>'
            else:
                res = '<span class="ret">(still running)</span>'
            title = f'{tag}<span class="fn">{e(n.name)}</span>({args}) <span class="arrow">→</span> {res}'
        cls = "call err" if n.raised else "call"
        is_open = " open" if depth == 0 else ""
        out.append(f'<details class="{cls}"{is_open}><summary>{title}'
                   f'<span class="meta">{e(self._rel(n.file))}:{n.lineno} · {dur}</span></summary><div class="body">')
        if n.args or n.globs:
            out.append('<table class="io"><tr><th colspan=2>INPUTS</th></tr>')
            for k, _, h in n.args:
                out.append(f"<tr><td class=k>{e(k)}</td><td>{h}</td></tr>")
            for k, h in n.globs:
                out.append(f'<tr><td class=k>{e(k)} <span class="gl">global</span></td><td>{h}</td></tr>')
            out.append("</table>")
        for it in n.items:
            kind = it[0]
            if kind == "call":
                self._render(it[1], out, depth + 1)
            elif kind == "line":
                _, no, src, ch = it
                chs = "".join(f'<div class="chg">{e(k)} = {e(v)}</div>' for k, v in ch)
                out.append(f'<div class="ln"><span class="no">{no}</span><code>{e(src)}</code>{chs}</div>')
            elif kind == "exc":
                out.append(f'<div class="exc">⚠ {e(it[1])}</div>')
        if n.skipped_lines:
            out.append(f'<div class="note">… {n.skipped_lines:,} more lines not shown (--max-lines)</div>')
        for fn_name, c in n.skipped_calls.items():
            out.append(f'<div class="note">… {c:,} more calls to {e(fn_name)}() not recorded (--max-calls)</div>')
        if n.name != "<module>":
            out.append('<table class="io out"><tr><th colspan=2>OUTPUT</th></tr>')
            if n.raised:
                out.append(f'<tr><td class=k>raised</td><td><code class="raise">{e(n.raised)}</code></td></tr>')
            elif n.ret:
                out.append(f"<tr><td class=k>return</td><td>{n.ret[1]}</td></tr>")
            if n.pkl:
                snippet = f'c = load_call(r"{n.pkl}")'
                out.append(f'<tr><td class=k>replay</td><td><code class="rp">{e(snippet)}</code>'
                           f'<button class="cp" onclick="cp(this)">copy</button></td></tr>')
            out.append("</table>")
        out.append("</div></details>")

    def write_html(self, path, elapsed, crash=None):
        sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
        body = []
        for _, n in self.roots:
            self._render(n, body, 0)
        crash_html = (f'<div class="crash">Script stopped with {html.escape(crash)}. '
                      f'The failing call is marked in red.</div>') if crash else ""
        with open(path, "w", encoding="utf-8") as f:
            f.write(_TEMPLATE.format(
                title=html.escape(os.path.basename(self.script)),
                when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                elapsed=f"{elapsed:.2f}", ncalls=len(self.index),
                data=html.escape(self.data_dir), crash=crash_html, body="\n".join(body)))


_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trace · {title}</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#86868b;--line:#e5e5ea;--fn:#0b5cad;--a:#6e3fb0;--ret:#1a7f37;--err:#c62828;--chg:#9a6700;--hl:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1c1e;--fg:#f2f2f7;--mut:#8e8e93;--line:#3a3a3c;--fn:#64a8ff;--a:#c79bff;--ret:#56d364;--err:#ff6b6b;--chg:#e3b341;--hl:#2c2c2e}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px 20px 60px}}
code,pre,.ln,summary{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0 10px;border-bottom:1px solid var(--line);margin-bottom:10px;z-index:2}}
h1{{font-size:17px;margin:0 0 4px}} .sub{{color:var(--mut);font-size:12px}}
button{{font:inherit;font-size:12px;padding:4px 10px;margin:6px 6px 0 0;border:1px solid var(--line);background:var(--hl);color:var(--fg);border-radius:6px;cursor:pointer}}
input{{font:inherit;font-size:12px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:180px}}
details.call{{border-left:2px solid var(--line);margin:3px 0 3px 4px;padding-left:10px}}
details.call>summary{{cursor:pointer;padding:3px 4px;border-radius:4px;list-style:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
details.call>summary::before{{content:"▸ ";color:var(--mut)}} details.call[open]>summary::before{{content:"▾ "}}
details.call>summary:hover{{background:var(--hl)}}
details.err{{border-left-color:var(--err)}}
.fn{{color:var(--fn);font-weight:600}} .a{{color:var(--a)}} .ret{{color:var(--ret)}} .arrow{{color:var(--mut)}}
.cid{{color:var(--mut);font-size:11px}} .gl{{color:var(--mut);font-size:10px;font-family:-apple-system,sans-serif}}
.raise{{color:var(--err);font-weight:600}} .meta{{color:var(--mut);font-size:11px;margin-left:12px}}
.body{{padding:2px 0 6px 6px}}
.ln{{padding:1px 0;white-space:pre-wrap}} .no{{display:inline-block;width:40px;color:var(--mut);text-align:right;margin-right:12px;user-select:none}}
.chg{{color:var(--chg);margin-left:52px;font-size:12px}}
.exc{{color:var(--err);margin-left:52px;font-size:12px}}
.note{{color:var(--mut);font-style:italic;margin:3px 0 3px 52px;font-size:12px}}
table.io{{border-collapse:collapse;margin:4px 0 6px 0;background:var(--hl);border-radius:6px;min-width:300px}}
table.io th{{text-align:left;font-size:10px;letter-spacing:.08em;color:var(--mut);padding:4px 8px 0}}
table.io td{{padding:2px 8px;vertical-align:top}} table.io td.k{{color:var(--a);font-family:ui-monospace,Menlo,monospace;font-size:13px}}
table.out th{{color:var(--ret)}}
code.rp{{font-size:11px;color:var(--mut);word-break:break-all}} button.cp{{margin:0 0 0 8px;padding:1px 8px;font-size:11px}}
details.val>summary{{cursor:pointer}} details.val>summary::marker{{color:var(--mut)}}
table.df{{border-collapse:collapse;margin:4px 0;font-size:12px}} table.df th,table.df td{{border:1px solid var(--line);padding:2px 6px;text-align:right}}
.dtypes{{color:var(--mut);font-size:11px}}
.crash{{background:var(--err);color:#fff;padding:8px 12px;border-radius:6px;margin:8px 0}}
body.compact .ln,body.compact .note,body.compact .exc{{display:none}}
.hit>summary{{outline:2px solid var(--chg)}}
</style></head><body>
<header><h1>Trace · {title}</h1>
<div class="sub">recorded {when} · {elapsed} s (with tracing overhead) · {ncalls} calls saved to {data} · click a call to open it</div>
<button onclick="setAll(true)">Expand all</button><button onclick="setAll(false)">Collapse all</button>
<button id="cb" onclick="toggleCompact()">Hide code lines (calls only)</button>
<input id="q" placeholder="find function…" oninput="find(this.value)">
</header>{crash}
<main>{body}</main>
<script>
function setAll(o){{document.querySelectorAll('details.call').forEach(d=>d.open=o)}}
function toggleCompact(){{const c=document.body.classList.toggle('compact');document.getElementById('cb').textContent=c?'Show code lines':'Hide code lines (calls only)'}}
function find(q){{document.querySelectorAll('.hit').forEach(d=>d.classList.remove('hit'));if(!q)return;let first=null;
 document.querySelectorAll('details.call').forEach(d=>{{const f=d.querySelector(':scope>summary .fn');if(f&&f.textContent.toLowerCase().includes(q.toLowerCase())){{d.classList.add('hit');first=first||d;let p=d.parentElement;while(p){{if(p.tagName==='DETAILS')p.open=true;p=p.parentElement}}}}}});
 if(first)first.scrollIntoView({{block:'center'}})}}
function cp(b){{const t=b.previousElementSibling.textContent;const done=()=>{{b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}};
 if(navigator.clipboard)navigator.clipboard.writeText(t).then(done,()=>sel(b));else sel(b)}}
function sel(b){{const r=document.createRange();r.selectNodeContents(b.previousElementSibling);const s=getSelection();s.removeAllRanges();s.addRange(r)}}
</script></body></html>"""


# --------------------------------------------------------------- replay API
_SCRIPT_MODS = {}


def _script_module(script):
    """The main script as an importable module, WITHOUT running its top-level code:
    only imports, function defs and class defs are executed."""
    import ast
    if script in _SCRIPT_MODS:
        return _SCRIPT_MODS[script]
    with open(script, encoding="utf-8") as f:
        tree = ast.parse(f.read(), script)
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    tree.body = [s for s in tree.body if isinstance(s, keep)]
    mod = types.ModuleType("_mydbg_script")
    mod.__file__ = script
    sys.modules["_mydbg_script"] = mod
    exec(compile(tree, script, "exec"), mod.__dict__)
    _SCRIPT_MODS[script] = mod
    return mod


class _Unpickler(pickle.Unpickler):
    """Classes defined in the main script were pickled as __main__.X - find them there."""

    def __init__(self, f, script):
        super().__init__(f)
        self.script = script

    def find_class(self, module, name):
        if module == "__main__":
            main = sys.modules.get("__main__")
            if not hasattr(main, name):
                return getattr(_script_module(self.script), name)
        return super().find_class(module, name)


class Call:
    """One saved call of one of your functions. See module docstring."""

    def __init__(self, path):
        import io
        self.path = os.path.abspath(path)
        with open(self.path, "rb") as f:
            d = pickle.load(f)
        self.meta = m = d["meta"]
        self._raw = d
        self._io = io
        for p in (m["root"], os.path.dirname(m["file"]), os.path.dirname(m["script"])):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.id, self.name, self.qualname = m["id"], m["name"], m["qualname"]
        self.file, self.line, self.raised, self.ms = m["file"], m["line"], m["raised"], m["ms"]
        self.inputs = self._load("inputs")
        self.globals = self._load("globals")
        self.output = None if d["output"] is None else self._load("output")["output"]

    def _load(self, key):
        return _Unpickler(self._io.BytesIO(self._raw[key]), self.meta["script"]).load()

    @property
    def func(self):
        m = self.meta
        if "<locals>" in m["qualname"]:
            raise RuntimeError(f"{m['qualname']} is a nested function - it cannot be re-imported.")
        if m["module"] in ("__main__", None) or os.path.abspath(m["file"]) == m["script"]:
            obj = _script_module(m["script"])
        else:
            obj = importlib.import_module(m["module"])
        for part in m["qualname"].split("."):
            obj = getattr(obj, part)
        return obj

    def _args(self, changes):
        m = self.meta
        inp = self._load("inputs")                      # fresh copy every time
        unknown = set(changes) - set(m["names"])
        if unknown:
            raise TypeError(f"{self.qualname}() has no input(s) {sorted(unknown)}; inputs are {m['names']}")
        inp.update(changes)
        names, ac, kc = m["names"], m["argcount"], m["kwonly"]
        args = [inp[n] for n in names[:ac]]
        kwargs = {n: inp[n] for n in names[ac:ac + kc] if n in inp}
        i = ac + kc
        if m["varargs"]:
            args += list(inp.get(names[i], ())); i += 1
        if m["varkw"]:
            kwargs.update(inp.get(names[i], {}))
        bad = [k for k, v in list(zip(names, args)) + list(kwargs.items()) if isinstance(v, Unsaved)]
        if bad:
            raise RuntimeError(f"input(s) {bad} could not be saved; pass them yourself: rerun({bad[0]}=...)")
        return args, kwargs

    def _with_globals(self, fn, run, override=None):
        g = fn.__globals__
        saved = self._load("globals")
        saved.update(override or {})
        old = {k: g.get(k, _MISSING) for k in saved}
        g.update({k: v for k, v in saved.items() if not isinstance(v, Unsaved)})
        try:
            return run()
        finally:
            for k, v in old.items():
                if v is _MISSING:
                    g.pop(k, None)
                else:
                    g[k] = v

    def rerun(self, _globals=None, **changes):
        """Run the function again on the saved inputs (and saved globals).
        Change inputs by name: rerun(lags=5).  Change globals: rerun(_globals={"SCENARIO": {...}})."""
        fn = self.func
        args, kwargs = self._args(changes)
        return self._with_globals(fn, lambda: fn(*args, **kwargs), _globals)

    def debug(self, _globals=None, **changes):
        """Step through the function in pdb (library code skipped) with the saved inputs."""
        fn = self.func
        args, kwargs = self._args(changes)
        dbg = ProjectPdb(self.meta["root"])
        return self._with_globals(fn, lambda: dbg.runcall(fn, *args, **kwargs), _globals)

    def check(self, _globals=None, **changes):
        """Rerun and compare with the saved output. Returns True / False and prints the difference."""
        if self.raised:
            try:
                self.rerun(_globals, **changes)
                print(f"Originally raised {self.raised}, now runs without error.\n" + _NOTE)
                return False
            except Exception as e:
                print(f"Raises again: {type(e).__name__}: {e}")
                return True
        new = self.rerun(_globals, **changes)
        ok, msg = _compare(self.output, new)
        print("Same output as recorded." if ok else f"DIFFERENT from recorded output:\n{msg}\n" + _NOTE)
        return ok

    def __repr__(self):
        ins = ", ".join(f"{k}={_short(v, 50)}" for k, v in self.inputs.items())
        gl = ", ".join(f"{k}={_short(v, 50)}" for k, v in self.globals.items()) or "(none)"
        out = f"raised {self.raised}" if self.raised else _short(self.output, 100)
        return (f"Call #{self.id}  {self.qualname}  ({os.path.basename(self.file)}:{self.line}, {self.ms} ms)\n"
                f"  inputs : {ins}\n  globals: {gl}\n  output : {out}\n"
                f"  -> .inputs  .globals  .output  .rerun(**changes)  .check()  .debug()")


_MISSING = object()
_NOTE = ("(If you did not change anything: the function also depends on something that is not saved -\n"
         " e.g. files it reads, sys.argv, random numbers without a seed, the current date, or code you edited since.)")


def _compare(a, b):
    try:
        import pandas as pd
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b); return True, ""
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            pd.testing.assert_series_equal(a, b); return True, ""
    except ImportError:
        pass
    except AssertionError as e:
        return False, str(e)
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            ok = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
            return bool(ok), "" if ok else f"recorded: {_short(a)}\nnow:      {_short(b)}"
    except (ImportError, TypeError):
        pass
    try:
        ok = bool(a == b)
    except Exception:
        ok = False
    return ok, "" if ok else f"recorded: {_short(a, 300)}\nnow:      {_short(b, 300)}"


def _resolve_dir(folder):
    if folder:
        return folder
    for cand in ("trace_calls",):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("No trace_calls folder here; pass folder=... (the *_calls folder next to the .html)")


def load_call(which, folder=None):
    """load_call(7) or load_call("trace_calls/00007_add_features.pkl")."""
    if isinstance(which, int):
        folder = _resolve_dir(folder)
        hits = [f for f in os.listdir(folder) if f.startswith(f"{which:05d}_") and f.endswith(".pkl")]
        if not hits:
            raise FileNotFoundError(f"call #{which} was not saved in {folder} (see list_calls())")
        which = os.path.join(folder, hits[0])
    return Call(which)


def list_calls(folder=None, function=None):
    """Table of saved calls (a DataFrame if pandas is installed). Filter by function name."""
    folder = _resolve_dir(folder)
    with open(os.path.join(folder, "index.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if function:
        rows = [r for r in rows if function in r["function"]]
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame(rows)
    except ImportError:
        return rows


# --------------------------------------------------------------------- main
def _main():
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("step", "record"):
        print(__doc__); sys.exit(1)
    mode = a.pop(0)
    opts = {"--out": "trace.html", "--root": None, "--max-calls": "20",
            "--max-lines": "2000", "--rows": "5", "--max-mb": "200"}
    while a and a[0] in opts:
        opts[a[0]] = a[1]; a = a[2:]
    script = os.path.abspath(a[0])
    root = opts["--root"] or os.path.dirname(script)

    # run the script as a real __main__ module so its classes/functions pickle by name
    sys.argv = a
    sys.path.insert(0, os.path.dirname(script))
    main_mod = types.ModuleType("__main__")
    main_mod.__file__ = script
    main_mod.__builtins__ = __builtins__
    sys.modules["__main__"] = main_mod
    glb = main_mod.__dict__
    with open(script, "rb") as f:
        code = compile(f.read(), script, "exec")

    if mode == "step":
        ProjectPdb(root).run(code, glb)
        return

    data_dir = os.path.splitext(opts["--out"])[0] + "_calls"
    os.makedirs(data_dir, exist_ok=True)
    for fn in os.listdir(data_dir):                      # clear the previous run
        if fn.endswith(".pkl") or fn == "index.csv":
            os.remove(os.path.join(data_dir, fn))
    rec = Recorder(root, script, data_dir, int(opts["--max-calls"]), int(opts["--max-lines"]),
                   int(opts["--rows"]), float(opts["--max-mb"]))
    t0, crash = time.perf_counter(), None
    sys.settrace(rec.global_trace)
    try:
        exec(code, glb)
    except BaseException as e:
        crash = f"{type(e).__name__}: {e}"
        raise
    finally:
        sys.settrace(None)
        rec.write_index()
        rec.write_html(opts["--out"], time.perf_counter() - t0, crash)
        print(f"[mydbg] report: {os.path.abspath(opts['--out'])}\n"
              f"[mydbg] {len(rec.index)} calls saved in {os.path.abspath(data_dir)}", file=sys.stderr)


if __name__ == "__main__":
    # Re-import this file as the module "mydbg", so saved objects never refer to "__main__.<our class>".
    _spec = importlib.util.spec_from_file_location("mydbg", _THIS)
    _m = importlib.util.module_from_spec(_spec)
    sys.modules["mydbg"] = _m
    _spec.loader.exec_module(_m)
    _m._main()
