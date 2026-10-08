"""
╔══════════════════════════════════════════════════════════╗
║          JARVIS - Offline Desktop Voice Assistant        ║
║                                                          ║
║  Requirements (install once):                            ║
║    pip install SpeechRecognition pyttsx3 pyaudio         ║
║                                                          ║
║  On Linux also run:                                      ║
║    sudo apt-get install python3-pyaudio portaudio19-dev  ║
║                                                          ║
║  On macOS:                                               ║
║    brew install portaudio                                ║
╚══════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import subprocess
import os
import shutil
import platform
import datetime
import time
import json
import re
import sys
import webbrowser
from pathlib import Path


# ─────────────────────────────────────────────
#  Try importing optional speech libraries
# ─────────────────────────────────────────────
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ─────────────────────────────────────────────
#  App registry  (add / edit freely)
# ─────────────────────────────────────────────
DEFAULT_APPS = {
    "notepad":     {"windows": "notepad.exe",          "linux": "gedit",          "darwin": "open -a TextEdit"},
    "calculator":  {"windows": "calc.exe",             "linux": "gnome-calculator","darwin": "open -a Calculator"},
    "browser":     {"windows": "start chrome",         "linux": "xdg-open http://","darwin": "open -a 'Google Chrome'"},
    "explorer":    {"windows": "explorer.exe",         "linux": "nautilus",       "darwin": "open ."},
    "terminal":    {"windows": "cmd.exe",              "linux": "gnome-terminal", "darwin": "open -a Terminal"},
    "paint":       {"windows": "mspaint.exe",          "linux": "pinta",          "darwin": "open -a Preview"},
    "word":        {"windows": "winword.exe",          "linux": "libreoffice --writer","darwin": "open -a 'Microsoft Word'"},
    "excel":       {"windows": "excel.exe",            "linux": "libreoffice --calc",  "darwin": "open -a 'Microsoft Excel'"},
    "powerpoint":  {"windows": "powerpnt.exe",         "linux": "libreoffice --impress","darwin": "open -a 'Microsoft PowerPoint'"},
    "vlc":         {"windows": "vlc.exe",              "linux": "vlc",            "darwin": "open -a VLC"},
    "vscode":      {"windows": "code",                 "linux": "code",           "darwin": "open -a 'Visual Studio Code'"},
    "spotify":     {"windows": "spotify.exe",          "linux": "spotify",        "darwin": "open -a Spotify"},
    "discord":     {"windows": "discord.exe",          "linux": "discord",        "darwin": "open -a Discord"},
    "task manager":{"windows": "taskmgr.exe",          "linux": "gnome-system-monitor","darwin": "open -a 'Activity Monitor'"},
}

CONFIG_FILE = Path.home() / ".jarvis_config.json"


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────
def get_os():
    s = platform.system().lower()
    if "win" in s:   return "windows"
    if "darwin" in s: return "darwin"
    return "linux"

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"apps": DEFAULT_APPS, "copy_jobs": [], "shortcuts": {}}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────
#  Core assistant logic
# ─────────────────────────────────────────────
class Assistant:
    def __init__(self, log_fn, status_fn):
        self.log     = log_fn
        self.status  = status_fn
        self.os      = get_os()
        self.config  = load_config()
        self.running = False

        # TTS engine
        self.tts = None
        if TTS_AVAILABLE:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate", 165)
            except Exception:
                pass

        # Speech recogniser
        self.recognizer = None
        self.mic        = None
        if SPEECH_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.pause_threshold = 0.8
                self.mic = sr.Microphone()
            except Exception:
                pass

    # ── speak ──────────────────────────────
    def speak(self, text):
        self.log(f"🤖 {text}")
        if self.tts:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception:
                pass

    # ── listen (one shot) ──────────────────
    def listen_once(self):
        if not self.recognizer or not self.mic:
            return None
        try:
            self.status("🎙 Listening…")
            with self.mic as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = self.recognizer.listen(src, timeout=5, phrase_time_limit=8)
            self.status("⚙ Processing…")
            text = self.recognizer.recognize_google(audio)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            self.log(f"[listen error] {e}")
            return None

    # ── process command ────────────────────
    def process(self, cmd):
        if not cmd:
            return
        cmd = cmd.lower().strip()
        self.log(f"🎤 You: {cmd}")

        # ── open app ──
        if re.search(r"\bopen\b", cmd):
            self._handle_open(cmd)

        # ── copy ──
        elif re.search(r"\bcopy\b", cmd):
            self._handle_copy(cmd)

        # ── time / date ──
        elif re.search(r"\btime\b", cmd):
            self.speak("The time is " + datetime.datetime.now().strftime("%I:%M %p"))

        elif re.search(r"\bdate\b", cmd):
            self.speak("Today is " + datetime.datetime.now().strftime("%A, %B %d %Y"))

        # ── system info ──
        elif re.search(r"\b(system|os|platform)\b", cmd):
            info = f"{platform.system()} {platform.release()}, Python {sys.version.split()[0]}"
            self.speak(info)

        # ── list files ──
        elif re.search(r"\blist\b.*\bfiles?\b", cmd):
            self._handle_list(cmd)

        # ── shutdown / exit ──
        elif re.search(r"\b(exit|quit|bye|stop)\b", cmd):
            self.speak("Goodbye!")
            return "EXIT"

        # ── volume (Windows only) ──
        elif re.search(r"\b(volume|mute)\b", cmd):
            self._handle_volume(cmd)

        # ── screenshot ──
        elif re.search(r"\bscreenshot\b", cmd):
            self._handle_screenshot()

        # ── help ──
        elif re.search(r"\bhelp\b", cmd):
            self._show_help()

        else:
            self.speak(f"I heard: {cmd}. I'm not sure how to handle that yet.")

    # ── open handler ───────────────────────
    def _handle_open(self, cmd):
        apps = self.config.get("apps", DEFAULT_APPS)
        for name, paths in apps.items():
            if name in cmd:
                executable = paths.get(self.os, "")
                if not executable:
                    self.speak(f"No path configured for {name} on {self.os}.")
                    return
                self.speak(f"Opening {name}")
                try:
                    if self.os == "windows":
                        os.startfile(executable) if Path(executable).exists() else subprocess.Popen(executable, shell=True)
                    else:
                        subprocess.Popen(executable.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    self.speak(f"Could not open {name}. {e}")
                return
        self.speak(f"I don't know how to open that. Add it in the App Manager tab.")

    # ── copy handler ───────────────────────
    def _handle_copy(self, cmd):
        # Expects pattern: copy <src_label> to <dst_label>
        # Labels are resolved from copy_jobs in config
        jobs = self.config.get("copy_jobs", [])
        for job in jobs:
            if job["name"].lower() in cmd:
                self._run_copy(job["src"], job["dst"])
                return
        # Fallback: ask user to pick via dialog
        self.speak("Please select the source folder.")
        src = filedialog.askdirectory(title="Select source folder")
        if not src:
            return
        self.speak("Now select the destination folder.")
        dst = filedialog.askdirectory(title="Select destination folder")
        if not dst:
            return
        self._run_copy(src, dst)

    def _run_copy(self, src, dst):
        src, dst = Path(src), Path(dst)
        if not src.exists():
            self.speak(f"Source not found: {src}"); return
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            if src.is_file():
                shutil.copy2(src, dst)
                count = 1
            else:
                for item in src.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst / item.name)
                        count += 1
                    elif item.is_dir():
                        shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                        count += 1
            self.speak(f"Copied {count} item{'s' if count != 1 else ''} to {dst.name}")
        except Exception as e:
            self.speak(f"Copy failed. {e}")

    # ── list files handler ─────────────────
    def _handle_list(self, cmd):
        folder = filedialog.askdirectory(title="Select folder to list")
        if not folder:
            return
        items = list(Path(folder).iterdir())
        names = [i.name for i in items[:15]]
        self.log("📂 " + ", ".join(names) + ("…" if len(items) > 15 else ""))
        self.speak(f"Found {len(items)} items in {Path(folder).name}")

    # ── volume handler ─────────────────────
    def _handle_volume(self, cmd):
        if self.os != "windows":
            self.speak("Volume control is only supported on Windows right now.")
            return
        if "mute" in cmd:
            subprocess.call(["nircmd.exe", "mutesysvolume", "1"])
            self.speak("Muted.")
        elif "up" in cmd or "increase" in cmd:
            subprocess.call(["nircmd.exe", "changesysvolume", "10000"])
            self.speak("Volume up.")
        elif "down" in cmd or "decrease" in cmd:
            subprocess.call(["nircmd.exe", "changesysvolume", "-10000"])
            self.speak("Volume down.")

    # ── screenshot ─────────────────────────
    def _handle_screenshot(self):
        try:
            import PIL.ImageGrab
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path.home() / f"screenshot_{ts}.png"
            PIL.ImageGrab.grab().save(out)
            self.speak(f"Screenshot saved to your home folder as screenshot_{ts}.png")
        except ImportError:
            self.speak("Pillow library not installed. Run: pip install Pillow")
        except Exception as e:
            self.speak(f"Screenshot failed. {e}")

    # ── help ───────────────────────────────
    def _show_help(self):
        self.speak("Here are some things you can say.")
        self.log(
            "📖 Commands:\n"
            "  open notepad / calculator / browser / vlc / vscode …\n"
            "  copy [job name]\n"
            "  list files\n"
            "  what time is it\n"
            "  what is the date\n"
            "  system info\n"
            "  screenshot\n"
            "  exit / quit / bye\n"
        )


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class JarvisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS — Offline Desktop Assistant")
        self.root.geometry("820x620")
        self.root.resizable(True, True)
        self.root.configure(bg="#0d1117")

        # colours
        C = {
            "bg":      "#0d1117",
            "panel":   "#161b22",
            "border":  "#30363d",
            "accent":  "#58a6ff",
            "green":   "#3fb950",
            "red":     "#f85149",
            "amber":   "#d29922",
            "text":    "#e6edf3",
            "muted":   "#8b949e",
            "btn_bg":  "#21262d",
        }
        self.C = C

        self._build_styles(C)
        self._build_ui(C)

        self.assistant = Assistant(
            log_fn    = self.log,
            status_fn = self.set_status,
        )

        self.listening = False
        self._welcome()

    # ── styles ────────────────────────────
    def _build_styles(self, C):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",           background=C["bg"],    borderwidth=0)
        style.configure("TNotebook.Tab",       background=C["panel"], foreground=C["muted"],
                        padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",             background=[("selected", C["bg"])],
                  foreground=[("selected", C["accent"])])
        style.configure("TFrame",              background=C["bg"])
        style.configure("TLabelframe",         background=C["bg"],    foreground=C["muted"],
                        bordercolor=C["border"])
        style.configure("TLabelframe.Label",   background=C["bg"],    foreground=C["muted"],
                        font=("Segoe UI", 9))
        style.configure("TLabel",              background=C["bg"],    foreground=C["text"],
                        font=("Segoe UI", 10))
        style.configure("Accent.TButton",      background=C["accent"], foreground="#0d1117",
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=[10,6])
        style.configure("Dark.TButton",        background=C["btn_bg"], foreground=C["text"],
                        font=("Segoe UI", 10), borderwidth=1, padding=[10,6])
        style.map("Accent.TButton",            background=[("active", "#79c0ff")])
        style.map("Dark.TButton",              background=[("active", "#30363d")])
        style.configure("TEntry",              fieldbackground=C["panel"], foreground=C["text"],
                        insertcolor=C["text"], bordercolor=C["border"], font=("Segoe UI", 11))
        style.configure("TScrollbar",          background=C["panel"], troughcolor=C["bg"],
                        arrowcolor=C["muted"])
        style.configure("Treeview",            background=C["panel"], foreground=C["text"],
                        fieldbackground=C["panel"], rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading",    background=C["btn_bg"], foreground=C["muted"],
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview",                  background=[("selected", C["accent"])],
                  foreground=[("selected", "#0d1117")])

    # ── UI scaffold ───────────────────────
    def _build_ui(self, C):
        # header
        hdr = tk.Frame(self.root, bg="#010409", height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  JARVIS", bg="#010409", fg=C["accent"],
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=18, pady=10)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self.status_var, bg="#010409", fg=C["muted"],
                 font=("Segoe UI", 10)).pack(side="right", padx=18)

        # notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._tab_main(nb, C)
        self._tab_apps(nb, C)
        self._tab_copy(nb, C)
        self._tab_text(nb, C)

    # ── Tab 1: Main ───────────────────────
    def _tab_main(self, nb, C):
        f = ttk.Frame(nb)
        nb.add(f, text="  Assistant  ")

        # log
        lf = ttk.LabelFrame(f, text=" Conversation log ")
        lf.pack(fill="both", expand=True, padx=12, pady=(12,6))
        self.log_box = scrolledtext.ScrolledText(
            lf, bg=C["panel"], fg=C["text"], insertbackground=C["text"],
            font=("Consolas", 10), wrap="word", relief="flat",
            borderwidth=0, state="disabled", height=18)
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_box.tag_config("bot",  foreground=C["accent"])
        self.log_box.tag_config("user", foreground=C["green"])
        self.log_box.tag_config("sys",  foreground=C["muted"])
        self.log_box.tag_config("err",  foreground=C["red"])

        # input row
        bot = ttk.Frame(f)
        bot.pack(fill="x", padx=12, pady=(0,10))
        self.cmd_var = tk.StringVar()
        entry = ttk.Entry(bot, textvariable=self.cmd_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        entry.bind("<Return>", lambda e: self._send_text())
        ttk.Button(bot, text="Send", style="Accent.TButton",
                   command=self._send_text).pack(side="left", padx=(6,0))

        # mic row
        mic_row = ttk.Frame(f)
        mic_row.pack(fill="x", padx=12, pady=(0,10))
        self.mic_btn = ttk.Button(mic_row, text="🎙  Hold to speak",
                                  style="Dark.TButton", command=self._toggle_listen)
        self.mic_btn.pack(side="left")
        self.mic_status = tk.Label(mic_row, text="", bg=C["bg"], fg=C["muted"],
                                   font=("Segoe UI", 9))
        self.mic_status.pack(side="left", padx=10)
        if not SPEECH_AVAILABLE:
            self.mic_status.config(text="⚠ Install SpeechRecognition + PyAudio for voice", fg=C["amber"])
        if not TTS_AVAILABLE:
            tk.Label(mic_row, text="⚠ Install pyttsx3 for voice output", bg=C["bg"],
                     fg=C["amber"], font=("Segoe UI", 9)).pack(side="left", padx=6)

    # ── Tab 2: App Manager ────────────────
    def _tab_apps(self, nb, C):
        f = ttk.Frame(nb)
        nb.add(f, text="  App Manager  ")

        cols = ("Name", "Windows", "Linux", "macOS")
        self.app_tree = ttk.Treeview(f, columns=cols, show="headings", height=14)
        for col in cols:
            self.app_tree.heading(col, text=col)
            self.app_tree.column(col, width=170 if col != "Name" else 120)
        self.app_tree.pack(fill="both", expand=True, padx=12, pady=(12,4))
        self._refresh_app_tree()

        # add / remove row
        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=(4,12))
        for lbl, var in [("Name", ""), ("Windows exe", ""), ("Linux cmd", ""), ("macOS cmd", "")]:
            tk.Label(row, text=lbl, bg=C["bg"], fg=C["muted"],
                     font=("Segoe UI", 9)).pack(side="left")
            e = ttk.Entry(row, width=14)
            e.pack(side="left", padx=(2, 8))
        # simple button-only version (we'll use dialogs to keep layout clean)
        ttk.Button(row, text="+ Add via dialog", style="Dark.TButton",
                   command=self._add_app_dialog).pack(side="left", padx=4)
        ttk.Button(row, text="✕ Remove selected", style="Dark.TButton",
                   command=self._remove_app).pack(side="left", padx=4)

    def _refresh_app_tree(self):
        self.app_tree.delete(*self.app_tree.get_children())
        apps = self.assistant.config.get("apps", DEFAULT_APPS) if hasattr(self, "assistant") else DEFAULT_APPS
        for name, paths in apps.items():
            self.app_tree.insert("", "end", values=(
                name,
                paths.get("windows", ""),
                paths.get("linux", ""),
                paths.get("darwin", ""),
            ))

    def _add_app_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Add App")
        dlg.configure(bg=self.C["bg"])
        dlg.geometry("380x260")
        fields = {}
        for i, (lbl, key) in enumerate([("App name (keyword)", "name"),
                                         ("Windows executable", "windows"),
                                         ("Linux command", "linux"),
                                         ("macOS command", "darwin")]):
            tk.Label(dlg, text=lbl, bg=self.C["bg"], fg=self.C["muted"],
                     font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w", padx=16, pady=8)
            e = ttk.Entry(dlg, width=28)
            e.grid(row=i, column=1, padx=8, pady=8)
            fields[key] = e

        def save():
            name = fields["name"].get().strip().lower()
            if not name:
                messagebox.showerror("Error", "App name required."); return
            self.assistant.config["apps"][name] = {
                "windows": fields["windows"].get().strip(),
                "linux":   fields["linux"].get().strip(),
                "darwin":  fields["darwin"].get().strip(),
            }
            save_config(self.assistant.config)
            self._refresh_app_tree()
            dlg.destroy()
            self.log(f"✅ App '{name}' added.")

        ttk.Button(dlg, text="Save", style="Accent.TButton", command=save).grid(
            row=4, column=0, columnspan=2, pady=14)

    def _remove_app(self):
        sel = self.app_tree.selection()
        if not sel:
            return
        name = self.app_tree.item(sel[0])["values"][0]
        self.assistant.config["apps"].pop(name, None)
        save_config(self.assistant.config)
        self._refresh_app_tree()
        self.log(f"🗑 App '{name}' removed.")

    # ── Tab 3: Copy Jobs ──────────────────
    def _tab_copy(self, nb, C):
        f = ttk.Frame(nb)
        nb.add(f, text="  File Manager  ")

        cols = ("Job name", "Source", "Destination")
        self.copy_tree = ttk.Treeview(f, columns=cols, show="headings", height=10)
        for col in cols:
            self.copy_tree.heading(col, text=col)
            self.copy_tree.column(col, width=220)
        self.copy_tree.pack(fill="both", expand=True, padx=12, pady=(12,4))
        self._refresh_copy_tree()

        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=6)
        ttk.Button(row, text="+ New copy job", style="Dark.TButton",
                   command=self._add_copy_dialog).pack(side="left", padx=4)
        ttk.Button(row, text="▶ Run selected", style="Accent.TButton",
                   command=self._run_selected_copy).pack(side="left", padx=4)
        ttk.Button(row, text="✕ Remove", style="Dark.TButton",
                   command=self._remove_copy).pack(side="left", padx=4)

        # quick copy section
        qf = ttk.LabelFrame(f, text=" Quick copy (pick folders now) ")
        qf.pack(fill="x", padx=12, pady=(8,12))
        self.src_var = tk.StringVar(value="(not set)")
        self.dst_var = tk.StringVar(value="(not set)")
        tk.Label(qf, textvariable=self.src_var, bg=C["bg"], fg=C["text"],
                 font=("Consolas", 9)).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        tk.Label(qf, textvariable=self.dst_var, bg=C["bg"], fg=C["text"],
                 font=("Consolas", 9)).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        ttk.Button(qf, text="Browse source", style="Dark.TButton",
                   command=lambda: self.src_var.set(filedialog.askdirectory() or self.src_var.get())
                   ).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        ttk.Button(qf, text="Browse dest", style="Dark.TButton",
                   command=lambda: self.dst_var.set(filedialog.askdirectory() or self.dst_var.get())
                   ).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        ttk.Button(qf, text="Copy now ▶", style="Accent.TButton",
                   command=self._quick_copy).grid(row=0, column=2, rowspan=2, padx=16)

    def _refresh_copy_tree(self):
        self.copy_tree.delete(*self.copy_tree.get_children())
        for job in self.assistant.config.get("copy_jobs", []) if hasattr(self, "assistant") else []:
            self.copy_tree.insert("", "end", values=(job["name"], job["src"], job["dst"]))

    def _add_copy_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("New Copy Job")
        dlg.configure(bg=self.C["bg"])
        dlg.geometry("400x200")
        tk.Label(dlg, text="Job name (voice keyword):", bg=self.C["bg"],
                 fg=self.C["muted"], font=("Segoe UI", 10)).grid(row=0, column=0, padx=16, pady=10)
        name_e = ttk.Entry(dlg, width=26)
        name_e.grid(row=0, column=1, padx=8)
        src_var = tk.StringVar(value="(not set)")
        dst_var = tk.StringVar(value="(not set)")
        ttk.Button(dlg, text="Source folder", style="Dark.TButton",
                   command=lambda: src_var.set(filedialog.askdirectory() or src_var.get())
                   ).grid(row=1, column=0, padx=16, pady=6)
        tk.Label(dlg, textvariable=src_var, bg=self.C["bg"], fg=self.C["text"],
                 font=("Consolas", 8), wraplength=200).grid(row=1, column=1)
        ttk.Button(dlg, text="Dest folder", style="Dark.TButton",
                   command=lambda: dst_var.set(filedialog.askdirectory() or dst_var.get())
                   ).grid(row=2, column=0, padx=16, pady=6)
        tk.Label(dlg, textvariable=dst_var, bg=self.C["bg"], fg=self.C["text"],
                 font=("Consolas", 8), wraplength=200).grid(row=2, column=1)

        def save():
            name = name_e.get().strip().lower()
            if not name:
                messagebox.showerror("Error", "Name required."); return
            self.assistant.config.setdefault("copy_jobs", []).append(
                {"name": name, "src": src_var.get(), "dst": dst_var.get()})
            save_config(self.assistant.config)
            self._refresh_copy_tree()
            dlg.destroy()

        ttk.Button(dlg, text="Save", style="Accent.TButton", command=save).grid(
            row=3, column=0, columnspan=2, pady=12)

    def _run_selected_copy(self):
        sel = self.copy_tree.selection()
        if not sel:
            return
        vals = self.copy_tree.item(sel[0])["values"]
        threading.Thread(target=self.assistant._run_copy,
                         args=(vals[1], vals[2]), daemon=True).start()

    def _remove_copy(self):
        sel = self.copy_tree.selection()
        if not sel:
            return
        name = self.copy_tree.item(sel[0])["values"][0]
        self.assistant.config["copy_jobs"] = [
            j for j in self.assistant.config.get("copy_jobs", []) if j["name"] != name]
        save_config(self.assistant.config)
        self._refresh_copy_tree()

    def _quick_copy(self):
        src, dst = self.src_var.get(), self.dst_var.get()
        if "(not set)" in (src, dst):
            messagebox.showerror("Error", "Set both source and destination first."); return
        threading.Thread(target=self.assistant._run_copy,
                         args=(src, dst), daemon=True).start()

    # ── Tab 4: Text command ───────────────
    def _tab_text(self, nb, C):
        f = ttk.Frame(nb)
        nb.add(f, text="  Commands  ")

        commands = [
            ("open <app>",      "Open a registered app"),
            ("copy <job>",      "Run a saved copy job"),
            ("list files",      "List files in a folder"),
            ("what time is it", "Speak current time"),
            ("what is the date","Speak today's date"),
            ("system info",     "OS & Python version"),
            ("screenshot",      "Capture screen (needs Pillow)"),
            ("help",            "Show all commands"),
            ("exit / quit",     "Close assistant"),
        ]
        tk.Label(f, text="Supported voice / text commands",
                 bg=C["bg"], fg=C["accent"], font=("Segoe UI", 13, "bold")).pack(
                 anchor="w", padx=16, pady=(16,8))
        for cmd, desc in commands:
            row = tk.Frame(f, bg=C["panel"], pady=6)
            row.pack(fill="x", padx=16, pady=2)
            tk.Label(row, text=f"  {cmd}", bg=C["panel"], fg=C["green"],
                     font=("Consolas", 11), width=26, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=C["panel"], fg=C["muted"],
                     font=("Segoe UI", 10)).pack(side="left", padx=8)

        tk.Label(f, text="Tip: register your own apps in the App Manager tab.",
                 bg=C["bg"], fg=C["muted"], font=("Segoe UI", 9)).pack(
                 anchor="w", padx=16, pady=12)

    # ── helpers ───────────────────────────
    def log(self, msg, tag="sys"):
        if "🤖" in msg:  tag = "bot"
        elif "🎤" in msg: tag = "user"
        elif "❌" in msg or "failed" in msg.lower(): tag = "err"
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line, tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _send_text(self):
        cmd = self.cmd_var.get().strip()
        if not cmd:
            return
        self.cmd_var.set("")
        threading.Thread(target=self.assistant.process, args=(cmd,), daemon=True).start()

    def _toggle_listen(self):
        if not SPEECH_AVAILABLE:
            messagebox.showwarning("Not available",
                "Install SpeechRecognition and PyAudio:\n\npip install SpeechRecognition pyaudio")
            return
        if self.listening:
            return
        self.listening = True
        self.mic_btn.config(text="🔴  Listening…")
        threading.Thread(target=self._listen_thread, daemon=True).start()

    def _listen_thread(self):
        result = self.assistant.listen_once()
        self.listening = False
        self.root.after(0, lambda: self.mic_btn.config(text="🎙  Hold to speak"))
        self.root.after(0, lambda: self.set_status("Ready"))
        if result:
            ret = self.assistant.process(result)
            if ret == "EXIT":
                self.root.after(500, self.root.destroy)
        else:
            self.log("(could not understand audio)", "sys")

    def _welcome(self):
        self.log("🤖 JARVIS online. Type a command or press the mic button.", "bot")
        if not SPEECH_AVAILABLE:
            self.log("⚠  Voice input unavailable — pip install SpeechRecognition pyaudio", "err")
        if not TTS_AVAILABLE:
            self.log("⚠  Voice output unavailable — pip install pyttsx3", "err")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = JarvisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()