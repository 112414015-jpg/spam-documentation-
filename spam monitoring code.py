import csv
import difflib
import logging
import os
import queue
import re
import smtplib
import socket
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, filedialog
from urllib.parse import urlparse

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
except ImportError:
    joblib = None

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "spam_model.joblib")
DATA_FILE = os.path.join(BASE, "spam.csv")
HISTORY_FILE = os.path.join(BASE, "scan_history.csv")

logging.basicConfig(
    filename=os.path.join(BASE, "alerts.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Small starter dataset. Add more labelled rows to spam.csv for better results.
SAMPLES = [
    ("ham", "Your college assignment is due tomorrow. Submit before 5 PM."),
    ("ham", "Your order has been shipped and will arrive on Friday."),
    ("ham", "Please find the meeting report attached."),
    ("ham", "Your OTP is 482913. Do not share it with anyone."),
    ("ham", "Can we reschedule tomorrow's class to 11 AM?"),
    ("ham", "Your electricity bill has been generated."),
    ("ham", "Thank you for your help. I will send the notes tonight."),
    ("ham", "Your internship interview is scheduled for Thursday."),
    ("ham", "Your package has been delivered."),
    ("ham", "Reminder: project review meeting at 3 PM."),
    ("ham", "The sale starts tomorrow. You can check the prices in the store."),
    ("ham", "Your account statement is ready to view in the official app."),
    ("spam", "Congratulations! You won a lottery. Send bank details to claim your prize."),
    ("spam", "URGENT! Your bank account is blocked. Click here and verify immediately."),
    ("spam", "You won a free iPhone. Click here to claim now."),
    ("spam", "Free money! Send your OTP and password to receive a cash reward."),
    ("spam", "Your credit card is suspended. Update details at http://secure-bank-verify.xyz"),
    ("spam", "Earn 50000 per week from home. No experience needed. Register now."),
    ("spam", "Your KYC is expired. Update now or your account will be closed."),
    ("spam", "Pay a processing fee to receive your lottery prize."),
    ("spam", "Send your password and OTP to avoid account suspension."),
    ("spam", "Your parcel is held at customs. Pay a fee at this link."),
    ("ham", "Hi, I have attached the notes from today's lecture."),
    ("ham", "The team meeting has been moved to Monday at 10 AM."),
    ("ham", "Please review the project document when you have time."),
    ("ham", "Your appointment is confirmed for 2:30 PM tomorrow."),
    ("ham", "Thank you for placing your order. Your receipt is attached."),
    ("ham", "The monthly electricity bill is available in your account."),
    ("ham", "I will call you after class this evening."),
    ("ham", "Your library book is due next Wednesday."),
    ("ham", "The exam timetable has been uploaded to the college portal."),
    ("ham", "Your payment receipt is available in the official application."),
    ("ham", "Please join the online meeting using the link shared by your teacher."),
    ("ham", "Your password was changed successfully. If this was not you, contact support."),
    ("ham", "The office will remain closed on the public holiday."),
    ("ham", "Can you send me the assignment draft before Friday?"),
    ("ham", "Your train ticket booking is confirmed."),
    ("ham", "The parcel you ordered has been delivered."),
    ("ham", "Your interview is scheduled for 11 AM. Please confirm your attendance."),
    ("ham", "The class recording is available on the student portal."),
    ("ham", "Your subscription renewal receipt is ready."),
    ("ham", "Please do not share your OTP with anyone."),
    ("spam", "Act now! Send your bank details to receive your guaranteed cash reward."),
    ("spam", "Your account is suspended. Click this link to restore access immediately."),
    ("spam", "You have been selected for a free gift. Pay a small fee to claim it."),
    ("spam", "Congratulations winner! Reply with your OTP to receive the prize."),
    ("spam", "Urgent: verify your password and credit card details at this website."),
    ("spam", "Earn unlimited money today. Register now and pay the joining fee."),
    ("spam", "Your KYC will expire today. Send your personal details to avoid closure."),
    ("spam", "Claim your lottery jackpot now by paying the processing charge."),
    ("spam", "We detected unusual activity. Click here and enter your banking password."),
    ("spam", "You won a smartphone. Share your OTP to confirm delivery."),
    ("spam", "Final warning: your bank account will be blocked unless you verify now."),
    ("spam", "Get a guaranteed loan instantly. Pay an advance fee to release the money."),
    ("spam", "You are today's lucky winner. Send your card number to claim the cash."),
    ("spam", "Your package is on hold. Pay customs charges through this unknown link."),
    ("spam", "Free reward waiting! Click here and provide your account password."),
    ("spam", "Your wallet is blocked. Verify your account details immediately."),
    ("spam", "Work from home and earn huge income. Pay registration fee now."),
    ("spam", "Congratulations, you have won a cash prize. Claim now."),
    ("spam", "Send your bank account number and OTP to receive your refund."),
    ("spam", "Your debit card is suspended. Update your details using this link.")
]

TEMP_DOMAINS = {
    "tempmail.com", "10minutemail.com", "mailinator.com",
    "guerrillamail.com", "yopmail.com", "throwawaymail.com"
}


def training_data():
    rows = list(SAMPLES)
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, encoding="utf-8-sig", newline="") as file:
                for row in csv.reader(file):
                    if len(row) >= 2 and row[0].strip().lower() in ("ham", "spam"):
                        rows.append((row[0].strip().lower(), row[1].strip()))
        except (OSError, UnicodeError, csv.Error):
            pass

    # Keep one label per unique message.
    unique = {}
    for label, message in rows:
        if message:
            unique[message.lower()] = (label, message)
    return list(unique.values())


def load_model():
    if joblib is None:
        raise RuntimeError(
            "Required packages are missing. Open Command Prompt and run:\n"
            "py -m pip install scikit-learn joblib"
        )

    # Rebuild the model on each run so old saved models do not keep outdated behaviour.
    rows = training_data()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    )
    model.fit([row[1] for row in rows], [row[0] for row in rows])

    try:
        joblib.dump(model, MODEL_FILE)
    except OSError:
        pass
    return model


def email_format(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    if (not re.fullmatch(pattern, email)
            or ".." in email
            or email.startswith(".")
            or "@" not in email
            or email.split("@", 1)[0].endswith(".")):
        return False, "Enter a valid email address, e.g. name@gmail.com."
    # Only Gmail addresses are accepted (anything after gmail.com, or another domain, is invalid).
    if email.rsplit("@", 1)[1].lower() != "gmail.com":
        return False, "Invalid email ID. Only @gmail.com addresses are accepted, e.g. name@gmail.com."
    return True, "Email address format is valid."


def domain_check(email):
    domain = email.rsplit("@", 1)[1].lower()
    if domain in TEMP_DOMAINS:
        return f"WARNING — {domain} is a known disposable email domain."
    try:
        socket.getaddrinfo(domain, None)
        return f"Domain {domain} resolves on the internet."
    except OSError:
        try:
            socket.getaddrinfo("google.com", None)
        except OSError:
            return "CANNOT VERIFY — internet connection or DNS is unavailable."
        return f"WARNING — domain {domain} could not be resolved."


def domain_exists(email):
    """True/False if the sender's domain exists; None if we cannot tell (no internet)."""
    domain = email.rsplit("@", 1)[1].lower()
    if domain in TEMP_DOMAINS:
        return True  # disposable domains are handled separately as a warning
    try:
        socket.getaddrinfo(domain, None)
        return True
    except OSError:
        pass
    try:
        socket.getaddrinfo("google.com", None)  # is the internet working at all?
    except OSError:
        return None
    return False


def mailbox_check(email):
    """Do not claim a Gmail mailbox exists or does not exist.

    Gmail does not provide a dependable public mailbox-existence check.
    SMTP RCPT responses can be misleading or deliberately obscured, and
    many networks block outbound port 25. This app therefore reports
    CANNOT VERIFY rather than producing a false positive/negative.
    """
    return (
        "CANNOT VERIFY",
        "Gmail does not provide a reliable public check for whether this mailbox exists. "
        "The address format and domain can be checked, but mailbox ownership/existence "
        "cannot be confirmed by this app."
    )


def sender_check(email):
    """Check only the visible sender text; this does not authenticate the sender."""
    user, domain = email.lower().split("@", 1)
    reasons = []

    if domain in TEMP_DOMAINS:
        reasons.append("uses a disposable email domain")
    if any(word in user for word in (
        "verify", "secure", "support", "admin", "alert", "payment", "urgent"
    )):
        reasons.append("sender name contains a commonly impersonated term")
    if re.search(r"\d{5,}", user):
        reasons.append("sender name contains five or more consecutive digits")

    if reasons:
        return "SUSPICIOUS SENDER PATTERN", "; ".join(reasons)

    return (
        "NO SUSPICIOUS SENDER PATTERN",
        "No common suspicious sender-name pattern was detected. "
        "This does not prove that the sender or mailbox is genuine."
    )


def link_check(text):
    links = re.findall(r"(?:https?://|www\.)[^\s<>\"']+", text, re.I)
    if not links:
        return "No links found."

    results = []
    for link in links:
        link = link.rstrip(".,);!?]")
        url = "http://" + link if link.lower().startswith("www.") else link
        parsed = urlparse(url)
        host = parsed.hostname
        flags = []
        if parsed.username or parsed.password or "@" in parsed.netloc:
            flags.append("contains an unusual @ sign")
        if host and host.replace(".", "").isdigit():
            flags.append("uses an IP address instead of a domain name")
        if host and host.lower().startswith("xn--"):
            flags.append("uses an internationalised/punycode domain")
        if host and host.lower() in {
            "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly"
        }:
            flags.append("uses a URL-shortening service")
        if url.lower().startswith("http://"):
            flags.append("does not use HTTPS")
        try:
            if not host:
                raise OSError()
            socket.getaddrinfo(host, None)
            status = "domain resolves"
        except OSError:
            status = "domain could not be resolved"
        note = "; ".join(flags) if flags else "no obvious URL pattern warning"
        results.append(f"{link}: {status}; {note} (not a safety guarantee)")
    return " | ".join(results)


def system_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "Not Available"
    finally:
        sock.close()


# Common spelling mistakes people make in spam-like words.
MISSPELLINGS = {
    "lotery": "lottery", "loterry": "lottery", "lottary": "lottery", "lotary": "lottery",
    "winer": "winner", "wiinner": "winner", "winnar": "winner",
    "prise": "prize", "prizee": "prize", "pasword": "password", "passwrd": "password",
    "congrats": "congratulations", "congratulation": "congratulations",
    "clik": "click", "clck": "click", "cliick": "click",
    "verfy": "verify", "varify": "verify", "acount": "account", "accout": "account",
    "blokked": "blocked", "suspendd": "suspended", "rewrd": "reward",
}
FUZZY_WORDS = [
    "lottery", "winner", "prize", "congratulations", "password", "suspended",
    "reward", "verify", "immediately", "jackpot", "account", "blocked", "claim"
]
PRIZE_WORDS = {"winner", "won", "lottery", "prize", "jackpot", "reward", "parisu"}
ACTION_WORDS = {"click", "claim", "send", "pay", "register", "reply", "call", "link"}


def normalise_text(text):
    """Fix common spelling mistakes so 'lotery winer' is read as 'lottery winner'."""
    def fix(match):
        word = match.group(0)
        if word in MISSPELLINGS:
            return MISSPELLINGS[word]
        if len(word) >= 5 and word not in FUZZY_WORDS:
            close = difflib.get_close_matches(word, FUZZY_WORDS, n=1, cutoff=0.85)
            if close:
                return close[0]
        return word
    return re.sub(r"[a-z]+", fix, text.lower())


def risk_label(risk):
    if risk >= 75:
        return "HIGH RISK"
    if risk >= 45:
        return "SPAM"
    if risk >= 20:
        return "SUSPICIOUS"
    return "NORMAL"


def analyse(model, subject, message):
    """
    Analyse subject and body only.
    Sender address and mailbox verification are shown separately.
    Clear content rules drive the result so ordinary words such as 'offer',
    'sale', 'OTP' or 'password' alone do not automatically mean spam.
    """
    text = normalise_text(subject + " " + message)
    matched = []
    rule_score = 0

    strong_phrases = (
        "lottery winner", "won a lottery", "you won", "claim your prize",
        "claim now", "free prize", "free money", "cash reward",
        "send your otp", "share your otp", "give your otp",
        "send otp", "share otp", "send your password",
        "share your password", "bank details", "credit card details",
        "account is blocked", "account blocked",
        "your account has been blocked", "your account is blocked",
        "account suspended", "your account has been suspended",
        "verify your account", "pay a processing fee", "processing fee",
        "account will be closed", "avoid account suspension",
        "lottery winner", "you are a winner", "you are winner", "you are the winner",
        "you have won", "you have been selected", "otp anuppunga", "otp kudunga",
        "kyc is expired", "kyc expired", "update your kyc", "kyc update"
    )
    medium_phrases = (
        "click here", "limited time", "act now", "verify immediately",
        "update your account", "lottery", "congratulations",
        "earn money", "work from home", "urgent payment",
        "free iphone", "cash prize", "register now",
        "winner", "prize", "jackpot", "congratulations", "gift card",
        "parisu", "link click", "click pannunga",
        "security alert", "unusual activity", "confirm your identity"
    )
    weak_phrases = ("urgent", "immediately", "hurry", "last chance", "fraud")

    for phrase in strong_phrases:
        if phrase in text and not any(phrase in m for m in matched):
            rule_score += 35
            matched.append(phrase)

    for phrase in medium_phrases:
        # Skip a phrase already covered by a longer matched one (no double counting).
        if phrase in text and not any(phrase in m for m in matched):
            rule_score += 15
            matched.append(phrase)

    for phrase in weak_phrases:
        if phrase in text and not any(phrase in m for m in matched):
            rule_score += 5
            matched.append(phrase)

    # Prize/winner word + an action word (click/claim/send/pay) is a classic scam pattern.
    words = set(re.findall(r"[a-z]+", text))
    if words & PRIZE_WORDS and words & ACTION_WORDS:
        rule_score += 25
        matched.append("prize/winner wording together with a call to action")

    urls = re.findall(r"(?:https?://|www\.)[^\s<>\"']+", text, re.I)
    if urls:
        rule_score += 15
        matched.append("contains a link")

    if len(re.findall(r"!", text)) >= 3:
        rule_score += 5
        matched.append("multiple exclamation marks")

    rule_score = min(rule_score, 100)

    # The ML model is supporting information only. It cannot turn a clean,
    # indicator-free message into spam by itself.
    try:
        model_text = "Subject: " + subject + "\nMessage: " + message
        probabilities = model.predict_proba([model_text])[0]
        spam_probability = float(
            probabilities[list(model.classes_).index("spam")]
        )
    except Exception:
        spam_probability = 0.0

    if rule_score == 0:
        risk = 5.0
    else:
        risk = min(100.0, rule_score * 0.85 + spam_probability * 100 * 0.15)

    return risk_label(risk), round(risk, 1), matched


class SpamApp:
    BG, WHITE, NAVY = "#f1f5f9", "#ffffff", "#0f172a"
    COLORS = {
        "NORMAL": ("#16a34a", "#dcfce7"),
        "SUSPICIOUS": ("#ea580c", "#ffedd5"),
        "SPAM": ("#dc2626", "#fee2e2"),
        "HIGH RISK": ("#991b1b", "#fecaca"),
        "INVALID EMAIL ID": ("#b45309", "#fef3c7")
    }

    def __init__(self, root):
        self.root = root
        root.title("Spam Monitoring and Alert System")
        root.geometry("900x800")
        root.configure(bg=self.BG)
        self.history = []
        self.model = None
        self.build_ui()
        # Old saved scans are intentionally NOT shown; history shows only scans from this session.
        self.status = tk.Label(root, text="Loading model...", bg=self.BG, fg="#64748b")
        self.status.pack(fill="x", padx=22, pady=5)
        self.check_btn.config(state="disabled")
        self.q = queue.Queue()  # worker thread sends results here; only the main thread touches Tk
        root.after(100, self.prepare_model)
        root.after(100, self.poll)

    def card(self, title):
        frame = tk.Frame(
            self.content, bg=self.WHITE, padx=16, pady=10,
            highlightbackground="#e2e8f0", highlightthickness=1
        )
        frame.pack(fill="x", padx=22, pady=(12, 0))
        tk.Label(
            frame, text=title, bg=self.WHITE, fg=self.NAVY,
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        return frame

    def build_ui(self):
        header = tk.Frame(self.root, bg=self.NAVY, height=58)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header, text="Spam Monitoring and Alert System", bg=self.NAVY,
            fg="white", font=("Segoe UI", 16, "bold")
        ).pack(side="left", padx=22)

        # Scrollable body (only addition: lets you scroll down to see everything)
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=self.BG, highlightthickness=0)
        vbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = tk.Frame(self.canvas, bg=self.BG)
        win = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width)
        )
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)

        box = self.card("Check an Email")
        self.sender = self.add_entry(box, "Sender Email ID")
        self.subject = self.add_entry(box, "Email Subject")
        tk.Label(
            box, text="Email Message / Body", bg=self.WHITE, fg="#64748b"
        ).pack(anchor="w")
        self.message = tk.Text(box, height=5, wrap="word", font=("Segoe UI", 10))
        self.message.pack(fill="x", pady=(3, 8))

        row = tk.Frame(box, bg=self.WHITE)
        row.pack(anchor="w")
        self.check_btn = ttk.Button(row, text="Check Email", command=self.check)
        self.check_btn.pack(side="left")
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=8)

        result = self.card("Detection Result")
        self.badge = tk.Label(
            result, text="No result yet", bg="#e2e8f0", fg="#64748b",
            font=("Segoe UI", 14, "bold"), padx=15, pady=6
        )
        self.badge.pack(anchor="w")
        self.details = tk.Label(
            result, text="", bg=self.WHITE, fg=self.NAVY,
            justify="left", wraplength=820
        )
        self.details.pack(anchor="w", pady=7)
        self.alert = tk.Label(
            result, text="", bg=self.WHITE, font=("Segoe UI", 10, "bold"),
            justify="left", wraplength=820
        )
        self.alert.pack(anchor="w")

        history = self.card("Scan History")
        ttk.Button(history, text="Export CSV", command=self.export).pack(anchor="e")
        table_box = tk.Frame(history, bg=self.WHITE)
        table_box.pack(fill="both", expand=True, pady=5)
        self.table = ttk.Treeview(
            table_box, columns=("time", "sender", "result", "confidence"),
            show="headings", height=5
        )
        table_bar = ttk.Scrollbar(table_box, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_bar.set)
        for col, width in (
            ("time", 145), ("sender", 280), ("result", 120), ("confidence", 100)
        ):
            self.table.heading(col, text=col.capitalize())
            self.table.column(col, width=width)
        table_bar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)

    def on_mousewheel(self, event):
        # scroll only when the content is taller than the visible area
        if self.content.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def add_entry(self, parent, title):
        tk.Label(
            parent, text=title, bg=self.WHITE, fg="#64748b"
        ).pack(anchor="w", pady=(5, 0))
        entry = ttk.Entry(parent)
        entry.pack(fill="x", pady=(2, 4))
        return entry

    def prepare_model(self):
        try:
            self.model = load_model()
            self.status.config(text="Ready")
            self.check_btn.config(state="normal")
        except Exception as exc:
            self.status.config(text="Model could not be loaded")
            messagebox.showerror("Setup Error", str(exc))

    def check(self):
        sender = self.sender.get().strip()
        subject = self.subject.get().strip()
        message = self.message.get("1.0", tk.END).strip()

        if not all((sender, subject, message)):
            messagebox.showwarning(
                "Missing Details", "Enter sender email, subject and message."
            )
            return

        valid, reason = email_format(sender)
        if not valid:
            self.display("INVALID EMAIL ID", reason, "Message was not analysed.")
            self.save(sender, "INVALID EMAIL ID", "-", subject, message)
            return

        if self.model is None:
            messagebox.showerror("Model Not Ready", "Please wait until the model is ready.")
            return

        self.status.config(text="Analysing... checking sender email and message (can take ~10 seconds)")
        self.check_btn.config(state="disabled")
        # Network checks (Gmail mailbox, DNS, links) can be slow, so run them off the UI thread.
        threading.Thread(target=self.worker, args=(sender, subject, message), daemon=True).start()

    def worker(self, sender, subject, message):
        try:
            if domain_exists(sender) is False:
                domain = sender.rsplit("@", 1)[1]
                reason = (f"The domain '{domain}' does not exist, so this email ID is not valid. "
                          "Please enter a correct email ID, e.g. name@gmail.com.")
                self.q.put(("invalid", sender, subject, message, reason))
                return
            data = {
                "sender_check": sender_check(sender),
                "mailbox": mailbox_check(sender),
                "domain": domain_check(sender),
                "links": link_check(subject + " " + message),
                # IMPORTANT: the message text decides the base spam risk.
                "analysis": analyse(self.model, subject, message),
                "ip": system_ip(),
            }
            self.q.put(("scan", sender, subject, message, data, None))
        except Exception as exc:
            self.q.put(("scan", sender, subject, message, None, exc))

    def poll(self):
        try:
            while True:
                kind, *args = self.q.get_nowait()
                if kind == "scan":
                    self.finish(*args)
                elif kind == "invalid":
                    self.finish_invalid(*args)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish_invalid(self, sender, subject, message, reason):
        self.check_btn.config(state="normal")
        self.display("INVALID EMAIL ID", reason, "Message was not analysed. Please enter a correct email ID.")
        self.save(sender, "INVALID EMAIL ID", "-", subject, message)
        messagebox.showwarning("Invalid Email ID", reason)

    def finish(self, sender, subject, message, data, error):
        self.check_btn.config(state="normal")
        if error is not None:
            self.status.config(text="Scan failed")
            messagebox.showerror("Scan Error", str(error))
            return

        sender_label, sender_info = data["sender_check"]
        mailbox_label, mailbox_info = data["mailbox"]
        domain_info = data["domain"]
        links = data["links"]
        _, content_risk, matched = data["analysis"]

        # Final risk = message content risk + penalties when the sender looks fake.
        risk = content_risk
        adjustments = []
        # Mailbox existence is not reliably verifiable through public Gmail SMTP.
        # Never increase spam risk based only on an uncertain mailbox response.
        if domain_info.startswith("WARNING"):
            risk += 20
            adjustments.append("sender domain problem +20%")
        if sender_label == "SUSPICIOUS SENDER PATTERN":
            risk += 10
            adjustments.append("suspicious sender pattern +10%")
        risk = round(min(risk, 100.0), 1)
        label = risk_label(risk)

        indicators = (", ".join(matched) if matched
                      else "No spam-related indicators found in the subject or message.")

        details = (
            "SENDER EMAIL CHECK (is this email ID real?)\n"
            f"  Sender: {sender}\n"
            f"  Domain: {domain_info}\n"
            f"  Sender pattern: {sender_label}\n\n"
            "MESSAGE CHECK (subject + body)\n"
            f"  Spam indicators: {indicators}\n"
            f"  Links: {links}\n"
            f"  Message spam risk: {content_risk}%\n\n"
            "FINAL DECISION\n"
            f"  Adjustments: {', '.join(adjustments) if adjustments else 'none'}\n"
            f"  Estimated spam risk: {risk}% (estimate, not a guarantee)\n"
            f"  System IP: {data['ip']}"
        )

        alerts = {
            "NORMAL": "NORMAL EMAIL - No clear spam indicators were detected.",
            "SUSPICIOUS": "CAUTION - Some suspicious indicators were detected. Verify before responding.",
            "SPAM": "SPAM ALERT - This email has multiple spam indicators. Do not click links or share details.",
            "HIGH RISK": "HIGH RISK - Strong scam/phishing indicators. Do not click links or share personal information."
        }

        self.display(label, details, alerts[label])

        if label in ("SPAM", "SUSPICIOUS", "HIGH RISK"):
            logging.warning("%s email from %s (%.1f%% risk)", label, sender, risk)

        self.save(sender, label, f"{risk}% risk", subject, message)
        messagebox.showinfo("Email Scan Result", f"{label}\n\n{details}")

    def display(self, label, details, alert):
        fg, bg = self.COLORS[label]
        self.badge.config(text=label, fg=fg, bg=bg)
        self.details.config(text=details)
        self.alert.config(text=alert, fg=fg)
        self.status.config(text=f"Last scan: {datetime.now():%H:%M:%S}")

    def load_history(self):
        """Load earlier scans so history remains available after closing the app."""
        if not os.path.exists(HISTORY_FILE):
            return
        try:
            with open(HISTORY_FILE, "r", newline="", encoding="utf-8-sig") as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 6 and row[0].strip().lower() != "time":
                        item = tuple(row[:6])
                        self.history.append(item)
                        self.table.insert("", 0, values=item[:4])
        except (OSError, UnicodeError, csv.Error) as exc:
            logging.error("Could not load scan history: %s", exc)

    def refresh_table(self):
        """Redraw the history table from self.history, newest scan on top."""
        self.table.delete(*self.table.get_children())
        for item in reversed(self.history):
            self.table.insert("", "end", values=item[:4])
        children = self.table.get_children()
        if children:
            self.table.selection_set(children[0])  # highlight newest scan
            self.table.see(children[0])
        self.root.update()  # force a repaint now, before any popup blocks the UI

    def save(self, sender, result, confidence, subject, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item = (now, sender, result, confidence, subject, message)
        self.history.append(item)
        self.refresh_table()
        try:
            file_exists = os.path.exists(HISTORY_FILE)
            with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exists or os.path.getsize(HISTORY_FILE) == 0:
                    writer.writerow([
                        "Time", "Sender", "Result", "Confidence", "Subject", "Message"
                    ])
                writer.writerow(item)
        except OSError as exc:
            logging.error("Could not save scan history: %s", exc)
            messagebox.showwarning(
                "History Save Warning",
                "This scan is shown now, but could not be saved permanently."
            )

    def clear(self):
        self.sender.delete(0, tk.END)
        self.subject.delete(0, tk.END)
        self.message.delete("1.0", tk.END)
        self.badge.config(text="No result yet", fg="#64748b", bg="#e2e8f0")
        self.details.config(text="")
        self.alert.config(text="")
        self.status.config(text="Ready")

    def export(self):
        if not self.history:
            messagebox.showinfo("Export", "No scan history to export.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Time", "Sender", "Result", "Confidence", "Subject", "Message"
                ])
                writer.writerows(self.history)
            messagebox.showinfo("Export", "History exported successfully.")


if __name__ == "__main__":
    window = tk.Tk()
    app = SpamApp(window)
    window.mainloop()
