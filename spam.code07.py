import csv
import logging
import os
import random
import re
import socket
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, filedialog

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    ML_ERROR = None
except ImportError as exc:
    ML_ERROR = str(exc)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "spam.csv")
LOG_PATH = os.path.join(BASE_DIR, "alerts.log")
SPAM_THRESHOLD = 0.50
SUSPICIOUS_THRESHOLD = 0.30   # NEW: 30% to below 50% spam probability -> SUSPICIOUS
HIGH_RISK_THRESHOLD = 0.80    # NEW: 80% and above spam probability   -> HIGH RISK

try:
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
except OSError:
    pass

HAND_DATA = [
    ('ham', 'Offer for Flipkart Big Billion Days starts tomorrow. Shop and save on electronics.'),
    ('ham', 'Flipkart: Your order #4587 has been shipped and will arrive on Friday.'),
    ('ham', 'Amazon sale is live now. Up to 40% off on selected items for Prime members.'),
    ('ham', 'Your Swiggy order is on the way. Delivery partner will reach in 10 minutes.'),
    ('ham', "Hi team please find the attached report for this week's meeting."),
    ('ham', "Can we reschedule tomorrow's class to 11 AM?"),
    ('ham', 'Your internship offer letter is attached. Please confirm your joining date.'),
    ('ham', 'Reminder: project review meeting at 3 PM in the conference room.'),
    ('ham', 'Thanks for your help yesterday. I will send the notes tonight.'),
    ('ham', 'Your electricity bill for October has been generated. Amount due is 540 rupees.'),
    ('ham', 'Happy birthday! Have a great year ahead.'),
    ('ham', 'Please submit the assignment before Monday evening.'),
    ('ham', 'Your Zomato order has been delivered. Rate your experience.'),
    ('ham', 'Hello sir I have completed the code. Please review it when you are free.'),
    ('ham', 'Myntra end of season sale is now live. Check out new arrivals.'),
    ('ham', 'Your OTP for login is 482913. Do not share it with anyone. Valid for 5 minutes.'),
    ('ham', 'Are you coming to college tomorrow?'),
    ('ham', 'Interview scheduled for Thursday 10 AM. Please carry your resume.'),
    ('ham', 'Your recharge of 299 rupees was successful. Validity 28 days.'),
    ('ham', 'Lunch at 1 PM today? Let me know where you want to go.'),
    ('ham', 'Meeting notes from today are shared in the drive folder.'),
    ('ham', 'Your package from Amazon has been delivered to your doorstep.'),
    ('ham', 'Diwali offer at our store. Visit us for discounts on all furniture.'),
    ('ham', 'Please find the invoice for your purchase attached with this mail.'),
    ('ham', 'Class timetable has been updated. Check the notice board.'),
    ('ham', 'Hey bro send me the project files when you get time.'),
    ('ham', 'Your application for the software internship has been received.'),
    ('ham', 'Good morning. Have a nice day.'),
    ('ham', 'Your bank statement for the month is ready. Log in to the official app to view it.'),
    ('ham', 'Congratulations on your promotion. Well deserved.'),
    ('spam', 'Congratulations! You have won a lottery of 50 lakh rupees. Claim your prize now by sending your bank details.'),
    ('spam', 'URGENT! Your bank account is blocked. Click this link and verify your account immediately.'),
    ('spam', 'You are the lucky winner of a free iPhone. Click here to claim now.'),
    ('spam', 'Free money! Send your OTP and password to receive 10000 rupees cash reward.'),
    ('spam', 'Dear customer your credit card has been suspended. Update your details at http://secure-bank-verify.xyz'),
    ('spam', 'WINNER!! You have been selected for a 1000 pound prize. Call 09061701461 to claim.'),
    ('spam', 'Earn 50000 per week from home. No experience needed. Click the link and register now.'),
    ('spam', 'Your KYC is expired. Update now or your account will be closed. Click http://bit.ly/kyc-update'),
    ('spam', 'You have won a free vacation to Dubai. Reply with your card number to confirm.'),
    ('spam', 'Get a loan of 5 lakh in 5 minutes without documents. Apply now limited time only.'),
    ('spam', 'Hot singles in your area want to meet you. Click here now.'),
    ('spam', 'Claim your free gift card worth 500 dollars. Hurry offer ends today.'),
    ('spam', 'Your Netflix payment failed. Verify your account now at http://netflix-secure-login.top'),
    ('spam', 'Lottery winner notification. Pay processing fee to receive your prize money.'),
    ('spam', 'Send your password and OTP immediately to avoid account suspension.'),
    ('spam', 'Make money fast with crypto. Guaranteed 10x returns in 7 days. Invest now.'),
    ('spam', 'Dear user you won cash reward of 25000. Share your bank account number to claim.'),
    ('spam', 'Your parcel is held at customs. Pay 200 rupees at this link to release it.'),
    ('spam', 'Final notice: your electricity will be cut tonight. Call this number and pay immediately.'),
    ('spam', 'Free entry to win a brand new car. Text WIN to 80082 now.'),
    ('spam', 'Congratulations you are selected for a work from home job. Pay registration fee to start.'),
    ('spam', 'Increase your followers and earn free money. Click this link now.'),
    ('spam', 'Account blocked urgent action required. Verify your account details here.'),
    ('spam', 'Claim now! Your free prize is waiting. Limited time. Reply YES.'),
    ('spam', 'You have been pre approved for a credit card. Share OTP to activate.'),
    ('spam', 'Buy cheap medicines without prescription. No questions asked. Click here.'),
    ('spam', 'Your PayPal account is limited. Confirm your password at http://paypa1-secure.info'),
    ('spam', 'Win free cash prize daily. Join our lottery now and become a millionaire.'),
    ('spam', 'Urgent reply needed. Transfer money to claim your inheritance of 5 million dollars.'),
    ('spam', 'Get rich quick. Click this link to claim your free money now.'),
]


def generate_data(n=250):
    rnd = random.Random(42)
    pick = rnd.choice
    brands = ["Flipkart", "Amazon", "Myntra", "Swiggy", "Zomato", "Ajio", "Nykaa", "BigBasket", "Meesho"]
    when = ["tomorrow", "this Friday", "from Monday", "tonight at midnight", "this weekend"]
    items = ["electronics", "fashion", "groceries", "mobiles", "home appliances", "books"]
    status = ["shipped", "delivered", "confirmed", "dispatched", "packed"]
    days = ["Monday", "Tuesday", "Friday", "tomorrow", "end of the day", "next week"]
    docs = ["report", "assignment", "project files", "invoice", "meeting notes", "timetable", "resume"]
    times = ["10 AM", "11:30 AM", "2 PM", "4 PM", "9 AM"]
    prizes = ["a lottery of 50 lakh rupees", "a free iPhone", "10000 rupees cash reward",
              "a free vacation to Dubai", "a brand new car", "a 500 dollar gift card"]
    sens = ["bank details", "OTP", "password", "card number", "account number", "PAN number"]
    accts = ["bank account", "credit card", "Netflix account", "PayPal account", "KYC", "SIM card"]

    def link():
        return pick(["http://bit.ly/claim-%d" % rnd.randint(10, 99), "http://secure-verify.xyz",
                     "http://free-prize.top", "this link"])

    ham_t = [
        lambda: f"{pick(brands)} sale starts {pick(when)}. Shop and save on {pick(items)}.",
        lambda: f"Offer for {pick(brands)}",
        lambda: f"{pick(brands)} offer: get {pick([10, 20, 30, 40])} percent off on {pick(items)} this week.",
        lambda: f"Your {pick(brands)} order #{rnd.randint(1000, 9999)} has been {pick(status)}.",
        lambda: f"Please send the {pick(docs)} by {pick(days)}.",
        lambda: f"Meeting at {pick(times)} {pick(days)}. Please be on time.",
        lambda: f"Hi, can you share the {pick(docs)}? I will review it {pick(days)}.",
        lambda: f"Your interview is scheduled on {pick(days)} at {pick(times)}. Carry your {pick(docs)}.",
        lambda: f"Thanks for the {pick(docs)}. I will get back to you {pick(days)}.",
        lambda: f"Reminder: class at {pick(times)} {pick(days)}.",
        lambda: f"Your OTP is {rnd.randint(100000, 999999)}. Do not share it with anyone.",
        lambda: f"Happy {pick(['birthday', 'Diwali', 'Pongal', 'new year'])}! Have a great day.",
    ]
    spam_t = [
        lambda: f"Congratulations! You have won {pick(prizes)}. Claim now by sending your {pick(sens)}.",
        lambda: f"URGENT! Your {pick(accts)} is blocked. Click {link()} and verify immediately.",
        lambda: f"You are the lucky winner of {pick(prizes)}. Click here to claim now.",
        lambda: f"Dear customer your {pick(accts)} will be suspended today. Share your {pick(sens)} to avoid closure.",
        lambda: f"Earn {pick([20000, 50000, 100000])} per week from home. No experience needed. Register at {link()}",
        lambda: f"Free money! Send your {pick(sens)} to receive {pick(prizes)}.",
        lambda: f"Final notice: pay a small fee to release {pick(prizes)}. Reply now.",
        lambda: f"WINNER!! You were selected for {pick(prizes)}. Call now to claim. Limited time.",
        lambda: f"Get a loan of {pick([2, 5, 10])} lakh in 5 minutes without documents. Apply at {link()}",
        lambda: f"Your {pick(accts)} needs update. Failure to verify will lead to permanent block. {link()}",
    ]
    rows = [("ham", pick(ham_t)()) for _ in range(n)]
    rows += [("spam", pick(spam_t)()) for _ in range(n)]
    return rows


def load_training_rows():
    rows = list(HAND_DATA) + generate_data()
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, encoding="latin-1", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] in ("ham", "spam"):
                    rows.append((r[0], r[1]))
    seen, unique = set(), []
    for label, text in rows:
        if text not in seen:
            seen.add(text)
            unique.append((label, text))
    return unique


def train_model():
    rows = load_training_rows()
    labels = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
    )
    model.fit(texts, labels)
    joblib.dump(model, MODEL_PATH)
    return model


_model = None


def get_model():
    global _model
    if ML_ERROR:
        raise RuntimeError("scikit-learn is not installed.\n\nOpen Command Prompt and run:\n"
                           "pip install scikit-learn")
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = joblib.load(MODEL_PATH)
            except Exception:
                _model = None
        if _model is None:
            _model = train_model()
    return _model


GMAIL_USER_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)*$")


def validate_email(email):
    if not email:
        return False, "Email ID is empty."
    if re.search(r"\s", email):
        return False, "Email ID cannot contain spaces or extra words."
    if email != email.lower():                                   # CHANGED: capital letters not allowed
        return False, "Email ID must be in lowercase only. Capital letters are not allowed."
    if email.count("@") != 1:
        return False, "Email ID must contain exactly one @."
    user, domain = email.split("@")
    if domain != "gmail.com":                                    # CHANGED: removed .lower()
        return False, "Only Gmail IDs ending with @gmail.com are accepted."
    if not 6 <= len(user) <= 30:
        return False, "Gmail username must be 6 to 30 characters long."
    if not GMAIL_USER_PATTERN.match(user):                       # CHANGED: removed .lower()
        return False, "Gmail username can have only letters, numbers and single dots (not at the start or end)."
    return True, "Valid"


def classify(message):
    model = get_model()
    p_spam = float(model.predict_proba([message])[0][list(model.classes_).index("spam")])
    # NEW: four levels instead of two
    if p_spam >= HIGH_RISK_THRESHOLD:
        label, conf = "HIGH RISK", p_spam
    elif p_spam >= SPAM_THRESHOLD:
        label, conf = "SPAM", p_spam
    elif p_spam >= SUSPICIOUS_THRESHOLD:
        label, conf = "SUSPICIOUS", p_spam
    else:
        label, conf = "NORMAL", 1 - p_spam
    return {"label": label, "confidence": round(conf * 100, 1),
            "spam_probability": round(p_spam * 100, 1)}


def get_system_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "Not Available"
    finally:
        sock.close()


BG, CARD, NAVY = "#f1f5f9", "#ffffff", "#0f172a"
BLUE, RED, GREEN, GREY = "#2563eb", "#dc2626", "#16a34a", "#64748b"
ORANGE, ORANGE_BG = "#ea580c", "#ffedd5"      # NEW: SUSPICIOUS colours
DARKRED, DARKRED_BG = "#991b1b", "#fecaca"    # NEW: HIGH RISK colours
FONT = "Segoe UI"
history = []


class SpamApp:
    def __init__(self, root):
        self.root = root
        root.title("Spam Monitoring and Alert System")
        root.geometry("900x760")
        root.minsize(820, 700)
        root.configure(bg=BG)
        self._styles()
        self._header()
        self._input_card()
        self._result_card()
        self._history_card()
        self.status = tk.Label(root, text="Loading model... (first run trains it, please wait)",
                               bg=BG, fg=GREY, font=(FONT, 9), anchor="w")
        self.status.pack(fill="x", padx=24, pady=6)
        self.check_btn.state(["disabled"])
        root.after(300, self._load_model)

    def _load_model(self):
        try:
            get_model()
            self.status.config(text="Ready")
            self.check_btn.state(["!disabled"])
        except Exception as exc:
            self.status.config(text="Model could not be loaded")
            messagebox.showerror("Setup error", str(exc))

    def popup(self, label, fg, bg, body):
        win = tk.Toplevel(self.root)
        win.title("Detection Output")
        win.configure(bg=CARD)
        win.resizable(False, False)
        win.transient(self.root)
        tk.Label(win, text=label, bg=bg, fg=fg, font=(FONT, 20, "bold"),
                 padx=40, pady=14).pack(fill="x")
        tk.Label(win, text=body, bg=CARD, fg=NAVY, font=(FONT, 11), justify="left",
                 wraplength=420, padx=24, pady=16).pack()
        ttk.Button(win, text="OK", style="Primary.TButton", command=win.destroy).pack(pady=(0, 18))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.bind("<Return>", lambda e: win.destroy())
        win.grab_set()
        win.focus_set()

    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Primary.TButton", font=(FONT, 11, "bold"), background=BLUE,
                    foreground="white", padding=(18, 8), borderwidth=0)
        s.map("Primary.TButton", background=[("active", "#1d4ed8")])
        s.configure("Ghost.TButton", font=(FONT, 11), padding=(18, 8))
        s.configure("Treeview", font=(FONT, 10), rowheight=26)
        s.configure("Treeview.Heading", font=(FONT, 10, "bold"))

    def _header(self):
        bar = tk.Frame(self.root, bg=NAVY, height=70)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="Spam Monitoring and Alert System", bg=NAVY, fg="white",
                 font=(FONT, 18, "bold")).pack(side="left", padx=24)
        tk.Label(bar, text="AI-based message detection", bg=NAVY, fg="#94a3b8",
                 font=(FONT, 10)).pack(side="right", padx=24)

    def _card(self, title):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="x", padx=24, pady=(14, 0))
        card = tk.Frame(outer, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1)
        card.pack(fill="x")
        tk.Label(card, text=title, bg=CARD, fg=NAVY, font=(FONT, 12, "bold")).pack(
            anchor="w", padx=18, pady=(12, 6))
        return card

    def _input_card(self):
        card = self._card("Check a Message")
        tk.Label(card, text="Sender Email ID", bg=CARD, fg=GREY, font=(FONT, 10)).pack(anchor="w", padx=18)
        self.sender_box = ttk.Entry(card, font=(FONT, 11))
        self.sender_box.pack(fill="x", padx=18, pady=(2, 10))
        tk.Label(card, text="Email Message", bg=CARD, fg=GREY, font=(FONT, 10)).pack(anchor="w", padx=18)
        self.msg_box = tk.Text(card, height=6, font=(FONT, 11), relief="solid", bd=1, wrap="word")
        self.msg_box.pack(fill="x", padx=18, pady=(2, 10))
        row = tk.Frame(card, bg=CARD)
        row.pack(anchor="w", padx=18, pady=(0, 14))
        self.check_btn = ttk.Button(row, text="Check Message", style="Primary.TButton", command=self.check)
        self.check_btn.pack(side="left")
        ttk.Button(row, text="Clear", style="Ghost.TButton", command=self.clear).pack(side="left", padx=10)

    def _result_card(self):
        card = self._card("Detection Result")
        self.badge = tk.Label(card, text="No result yet", bg="#e2e8f0", fg=GREY,
                              font=(FONT, 16, "bold"), padx=20, pady=8)
        self.badge.pack(anchor="w", padx=18)
        self.detail = tk.Label(card, text="", bg=CARD, fg=NAVY, font=(FONT, 11),
                               justify="left", wraplength=800)
        self.detail.pack(anchor="w", padx=18, pady=(8, 2))
        self.alert = tk.Label(card, text="", bg=CARD, font=(FONT, 12, "bold"),
                              wraplength=800, justify="left")
        self.alert.pack(anchor="w", padx=18, pady=(0, 14))

    def _history_card(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(14, 0))
        card = tk.Frame(outer, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1)
        card.pack(fill="both", expand=True)
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x", padx=18, pady=(12, 6))
        tk.Label(top, text="Scan History", bg=CARD, fg=NAVY, font=(FONT, 12, "bold")).pack(side="left")
        ttk.Button(top, text="Export CSV", style="Ghost.TButton", command=self.export).pack(side="right")
        cols = ("time", "sender", "result", "confidence")
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (150, 300, 100, 110)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w")
        self.tree.tag_configure("SPAM", foreground=RED)
        self.tree.tag_configure("NORMAL", foreground=GREEN)
        self.tree.tag_configure("INVALID", foreground="#b45309")
        self.tree.tag_configure("SUSPICIOUS", foreground=ORANGE)    # NEW
        self.tree.tag_configure("HIGH RISK", foreground=DARKRED)     # NEW
        self.tree.pack(fill="both", expand=True, padx=18, pady=(0, 14))

    def check(self):
        try:
            self._check()
        except Exception as exc:
            messagebox.showerror("Error", f"Something went wrong:\n{exc}")

    def _check(self):
        sender = self.sender_box.get().strip()
        message = self.msg_box.get("1.0", tk.END).strip()
        if not sender or not message:
            messagebox.showwarning("Missing details", "Please enter the Email ID and the Email Message.")
            return

        ok, reason = validate_email(sender)
        if not ok:
            self.show("INVALID EMAIL ID", "#b45309", "#fef3c7", f"Sender: {sender}",
                      reason + " Message was not processed.")
            self.log(sender, "INVALID", "-", message)
            self.popup("INVALID EMAIL ID", "#b45309", "#fef3c7",
                       f"{reason}\n\nSender: {sender}\nMessage was not processed.")
            return

        self.status.config(text="Analysing message...")
        self.root.update_idletasks()
        r = classify(message)
        detail = (f"Sender: {sender}\nSystem IP: {get_system_ip()}\n"
                  f"Confidence: {r['confidence']}%   |   Spam probability: {r['spam_probability']}%")
        if r["label"] == "HIGH RISK":                                          # NEW
            self.show("HIGH RISK", DARKRED, DARKRED_BG, detail,
                      "ALERT: This message is HIGH RISK. Do not click links or share any details.")
            logging.error("HIGH RISK from %s (%.1f%%)", sender, r["spam_probability"])
        elif r["label"] == "SPAM":
            self.show("SPAM", RED, "#fee2e2", detail,
                      "ALERT: This message is SPAM. Do not click links or share details.")
            logging.warning("SPAM from %s (%.1f%%)", sender, r["spam_probability"])
        elif r["label"] == "SUSPICIOUS":                                       # NEW
            self.show("SUSPICIOUS", ORANGE, ORANGE_BG, detail,
                      "CAUTION: This message looks suspicious. Be careful with links and details.")
            logging.warning("SUSPICIOUS from %s (%.1f%%)", sender, r["spam_probability"])
        else:
            self.show("NORMAL", GREEN, "#dcfce7", detail, "This message looks safe.")
        self.log(sender, r["label"], f"{r['confidence']}%", message)
        if r["label"] == "HIGH RISK":                                          # NEW
            self.popup("HIGH RISK", DARKRED, DARKRED_BG,
                       f"This message is HIGH RISK.\nDo not click links or share any details.\n\n{detail}")
        elif r["label"] == "SPAM":
            self.popup("SPAM", RED, "#fee2e2",
                       f"This message is SPAM.\nDo not click links or share details.\n\n{detail}")
        elif r["label"] == "SUSPICIOUS":                                       # NEW
            self.popup("SUSPICIOUS", ORANGE, ORANGE_BG,
                       f"This message looks suspicious.\nBe careful with links and details.\n\n{detail}")
        else:
            self.popup("NORMAL", GREEN, "#dcfce7", f"This message looks safe.\n\n{detail}")

    def show(self, label, fg, bg, detail, alert):
        self.badge.config(text=label, fg=fg, bg=bg)
        self.detail.config(text=detail)
        self.alert.config(text=alert, fg=fg)
        self.status.config(text=f"Last scan: {datetime.now():%H:%M:%S}")

    def log(self, sender, result, conf, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append((now, sender, result, conf, message))
        self.tree.insert("", 0, values=(now, sender, result, conf), tags=(result,))

    def clear(self):
        self.sender_box.delete(0, tk.END)
        self.msg_box.delete("1.0", tk.END)
        self.badge.config(text="No result yet", fg=GREY, bg="#e2e8f0")
        self.detail.config(text="")
        self.alert.config(text="")

    def export(self):
        if not history:
            messagebox.showinfo("Export", "No scan history to export yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Time", "Sender", "Result", "Confidence", "Message"])
                w.writerows(history)
            messagebox.showinfo("Export", "History exported successfully.")


if __name__ == "__main__":
    root = tk.Tk()
    root.report_callback_exception = lambda *a: messagebox.showerror("Error", str(a[1]))
    SpamApp(root)
    root.mainloop()